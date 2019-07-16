import threading
import tensorflow as tf 
import numpy as np 
import json
import Nets 
import cv2

from Data_utils import preprocessing, weights_utils
from Sampler import sampler_factory
from Losses import loss_factory

class RealTimeStereo(threading.Thread):
    """
    Class that implement real time self adaptive stereo
    """
    _camera_window_name_left = "left frame"
    _camera_window_name_right = "right frame"
    _disparity_window_name = "disparity prediction"
    def __init__(
        self,
        camera_buffer,
        model_name='MADNet',
        weight_path=None,
        learning_rate=0.0001,
        block_config_path='../block_config/MadNet_full.json',
        image_shape=[480,640],
        crop_shape=[None,None],
        SSIMTh=0.5,
        mode='MAD'):
        """
        Create a self adaptive deep stereo system
        Args
        ---
        camera_buffer: queue
            queue to read left and right thread
        model_name: string
            name of the stereo CNN to create
        weight_path: string
            path to the initial weights for the Stereo CNN, None for random initialization
        learning_rate: float
            learning rate for online adaptation
        block_config_path: string
            path to the block_config json file
        image_shape: list of two int
            height and width of the stereo frames
        crop_shape: list of two int
            crop input to this dimension before feeding them to the network
        SSIMTh: float
            reset the network to the initial configuration if current SSIM > SSIMTh
        mode: string
            nline adaptation mode: NONE - perform only inference, FULL - full online backprop, MAD - backprop only on portions of the network
        """
        self._camera_buffer = camera_buffer
        self._model_name = model_name
        self._weight_path = weight_path
        self._learning_rate = learning_rate
        self._block_config_path = block_config_path
        self._image_shape = image_shape
        self._crop_shape = crop_shape
        self._SSIMTh = SSIMTh
        self._mode = mode 
        self._stop_flag = False
        self._ready = self._setup_graph()
        self._ready &= self._initialize_model()
        threading.Thread.__init__(self)

    def _load_block_config(self):
        #load json file config
        with open(self._block_config_path) as json_data:
            self._train_config = json.load(json_data)

    def _build_input_ops(self):
        #input placeholder ops
        self._left_placeholder = tf.placeholder(tf.float32,shape=[1,None, None,3], name='left_input')
        self._right_placeholder = tf.placeholder(tf.float32,shape=[1,None, None,3], name='right_input')

        self._left_input = self._left_placeholder
        self._right_input = self._right_placeholder
        
        if self._image_shape[0] is not None:
            self._left_input = preprocessing.rescale_image(self._left_input, self._image_shape)
            self._right_input = preprocessing.rescale_image(self._right_input, self._image_shape)
        
        if self._crop_shape[0] is not None:
            self._left_input = tf.image.resize_image_with_crop_or_pad(self._left_input, self._crop_shape[0], self._crop_shape[1])
            self._right_input = tf.image.resize_image_with_crop_or_pad(self._right_input, self._crop_shape[0], self._crop_shape[1])

    def _build_network(self):
        #network model
        with tf.variable_scope('model'):
            net_args = {}
            net_args['left_img'] = self._left_input
            net_args['right_img'] = self._right_input
            net_args['split_layers'] = [None]
            net_args['sequence'] = True
            net_args['train_portion'] = 'BEGIN'
            net_args['bulkhead'] = True if self._mode=='MAD' else False
            self._net = Nets.get_stereo_net(self._model_name, net_args)
            self._predictions = self._net.get_disparities()
            self._full_res_disp = self._predictions[-1]

            self._inputs = {
                'left':self._left_input,
                'right':self._right_input,
                'target':tf.zeros([1,self._image_shape[0],self._image_shape[1],1],dtype=tf.float32)
            }

            #full resolution loss between warped right image and original left image
            self._loss =  loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)(self._predictions,self._inputs)
    
    def _MAD_adaptation_ops(self):
        #build train ops for separate portions of the network
        self._load_block_config()

        #keep all predictions except full res
        predictions = self._predictions[:-1] 
        
        inputs_modules = self._inputs
        
        assert(len(predictions)==len(self._train_config))
        for counter,p in enumerate(predictions):
            print('Build train ops for disparity {}'.format(counter))
                    
            #rescale predictions to proper resolution
            multiplier = tf.cast(tf.shape(self._left_input)[1]//tf.shape(p)[1],tf.float32)
            p = preprocessing.resize_to_prediction(p,inputs_modules['left'])*multiplier

            #compute reprojection error
            with tf.variable_scope('reprojection_'+str(counter)):
                reconstruction_loss = loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)([p],inputs_modules)

            #build train op
            layer_to_train = self._train_config[counter]
            print('Going to train on {}'.format(layer_to_train))
            var_accumulator=[]
            for name in layer_to_train:
                var_accumulator+=self._net.get_variables(name)
            print('Number of variable to train: {}'.format(len(var_accumulator)))
                
            #add new training op
            self._train_ops.append(self._trainer.minimize(reconstruction_loss,var_list=var_accumulator))

            print('Done')
            print('='*50)
        
        #create Sampler to fetch portions to train
        self._sampler = sampler_factory.get_sampler('PROBABILITY',1,0)
    
    def _Full_adaptation_ops(self):
        self._train_ops.append(self._trainer.minimize(self._loss))
        self._sampler = sampler_factory.get_sampler('FIXED',1,0)
    
    def _no_adaptation_ops(self):
        #mock ops that don't do anything
        self._train_ops.append(tf.no_op())
        self._sampler = sampler_factory.get_sampler('FIXED',1,0)

    def _build_adaptation_ops(self):
        """
        Populate self._train_ops
        """
        #self._trainer = tf.train.MomentumOptimizer(self._learning_rate,0.9)
        self._trainer = tf.train.AdamOptimizer(self._learning_rate)
        self._train_ops = []
        if self._mode == 'MAD':
            self._MAD_adaptation_ops()
        elif self._mode == 'FULL':
            self._Full_adaptation_ops()
        elif self._mode == 'NONE':
            self._no_adaptation_ops()

    def _setup_graph(self):
        """
        Build tensorflow graph and ops
        """
        self._build_input_ops()

        self._build_network()

        self._build_adaptation_ops()

        return True
    
    def _initialize_model(self):
        """
        Create tensorflow session and initialize the network
        """
        #session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        #variable initialization
        initializers = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self._session.run(initializers)

        #restore disparity inference weights and populate self._restore_op
        if self._weight_path is not None:
            var_to_restore = weights_utils.get_var_to_restore_list(self._weight_path, [])
            self._restorer = tf.train.Saver(var_list=var_to_restore)
            self._restore_op = lambda: self._restorer.restore(self._session, self._weight_path)
            self._restore_op()
            print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))
        else:
            self._restore_op = lambda: self._session.run(initializers)

        #operation to select the different train ops
        num_actions=len(self._train_ops)
        self._sample_distribution=np.zeros(shape=[num_actions])
        self._temp_score = np.zeros(shape=[num_actions])
        self._loss_t_2 = 0
        self._loss_t_1 = 0
        self._expected_loss = 0
        self._last_trained_blocks = 0

        print('Network Ready')

        return True

    def _setup_gui(self):
        """
        Create a simple OpenCV GUI
        """
        cv2.namedWindow(self._camera_window_name_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self._camera_window_name_right, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self._disparity_window_name, cv2.WINDOW_NORMAL)
    
    def stop(self):
        """
        Stop the prediciton and kill the thread
        """
        self._stop_flag =True

    def run(self):
        self._setup_gui()
        first=True
        it=0
        while not self._stop_flag:
            #fetch frames
            frames = self._camera_buffer.get(block=True)
            left_frame = frames[:1]
            right_frame = frames[1:]

            #Fetch portion of network to train
            #softmax
            exp = np.exp(self._sample_distribution)
            distribution = exp/np.sum(exp,axis=0)
            train_op_id = self._sampler.sample(distribution)[0]
            selected_train_op = self._train_ops[train_op_id]

            #build list of tensorflow operations that needs to be executed + feed dict
            tf_fetches = [self._loss, selected_train_op, self._full_res_disp, self._left_input, self._right_input]
            fd = {
                self._left_placeholder: left_frame,
                self._right_placeholder: right_frame
            }

            #run network
            full_ssim, _, disp_prediction, lefty, righty = self._session.run(tf_fetches, feed_dict=fd)
            print('Step {}: {}'.format(it,full_ssim))

            if self._mode == 'MAD':
                #update sampling probabilities
                if first:
                    self._loss_t_2 = full_ssim
                    self._loss_t_1 = full_ssim
                self._expected_loss = 2*self._loss_t_1-self._loss_t_2	
                gain_loss=self._expected_loss-full_ssim
                self._sample_distribution = 0.99*self._sample_distribution
                self._sample_distribution[self._last_trained_blocks] += 0.01*gain_loss

                self._last_trained_blocks = train_op_id
                self._loss_t_2 = self._loss_t_1
                self._loss_t_1 = full_ssim
            
            if full_ssim>self._SSIMTh:
                print('Resetting Network...')
                self._restore_op()
            
            #show current detection
            cv2.imshow(self._camera_window_name_left, lefty[0].astype(np.uint8))
            cv2.imshow(self._camera_window_name_right, righty[0].astype(np.uint8))
            cv2.imshow(self._disparity_window_name, cv2.applyColorMap(disp_prediction[0].astype(np.uint8),cv2.COLORMAP_JET))
            cv2.waitKey(1)
            it+=1
        
        #close session
        self._session.close()
        cv2.destroyAllWindows()