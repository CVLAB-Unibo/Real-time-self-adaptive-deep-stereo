import tensorflow as tf

from Nets import Stereo_net
from Nets import sharedLayers
from Data_utils import preprocessing

MAX_DISP = 40

class DispNet(Stereo_net.StereoNet):
    _valid_args = [
        ("left_img", "meta op for left image batch"),
        ("right_img", "meta op for right image batch"),
        ("correlation", "flag to enable the use of the correlation layer")
    ] + Stereo_net.StereoNet._valid_args
    _netName = "Dispnet"

    def __init__(self, **kwargs):
        """
        Creation of a DispNet CNN
        """
        super(DispNet, self).__init__(**kwargs)

    def _validate_args(self, args):
        """
        Check that args contains everything that is needed
        Valid Keys for args:
            left_img: left image op
            right_img: right image op
            correlation: boolean, if True use correlation layer, defaults to True
        """
        super(DispNet, self)._validate_args(args)
        if ("left_img" not in args) or ("right_img" not in args):
            raise Exception('Missing input op for left and right images')
        if "correlation" not in args:
            print('WARNING: Correlation unspecified, setting to True')
            args['correlation'] = True
        return args

    def _make_disp(self,op):
        scale = self._left_input_batch.get_shape()[2].value/op.get_shape()[2].value
        op = tf.image.resize_images(tf.nn.relu(op*scale), [self._left_input_batch.get_shape()[1].value, self._left_input_batch.get_shape()[2].value])
        op = tf.image.resize_image_with_crop_or_pad(op, self._restore_shape[0], self._restore_shape[1])
        return op

    def _upsampling_block(self, bottom, skip_connection, input_channels, output_channels, skip_input_channels, name='upsample', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self._add_to_layers(name + '/deconv', sharedLayers.conv2d_transpose(
                bottom, [4, 4, output_channels, input_channels], strides=2, name='deconv'))
            self._add_to_layers(name + '/predict', sharedLayers.conv2d(bottom, [
                                3, 3, input_channels, 1], strides=1, activation=lambda x: x, name='predict'))
            self._disparities.append(self._make_disp(self._layers[name + '/predict']))
            self._add_to_layers(name + '/up_predict', sharedLayers.conv2d_transpose(self._get_layer_as_input(
                name + '/predict'), [4, 4, 1, 1], strides=2, activation=lambda x: x, name='up_predict'))
            with tf.variable_scope('join_skip'):    
                concat_inputs = tf.concat([skip_connection, self._get_layer_as_input(name + '/deconv'), self._get_layer_as_input(name + '/up_predict')], axis=3)
            self._add_to_layers(name + '/concat', sharedLayers.conv2d(concat_inputs, [
                                3, 3, output_channels + skip_input_channels + 1, output_channels], strides=1, activation=lambda x: x, name='concat'))

    def _preprocess_inputs(self, args):
        self._left_input_batch = args['left_img']
        self._restore_shape = tf.shape(args['left_img'])[1:3]
        self._left_input_batch = tf.cast(
            self._left_input_batch, dtype=tf.float32) / 255.0
        self._left_input_batch = self._left_input_batch - (100.0 / 255)
        self._left_input_batch = preprocessing.pad_image(
            self._left_input_batch, 64, dynamic=True)

        self._right_input_batch = args['right_img']
        self._right_input_batch = tf.cast(
            self._right_input_batch, dtype=tf.float32) / 255.0
        self._right_input_batch = self._right_input_batch - (100.0 / 255)
        self._right_input_batch = preprocessing.pad_image(
            self._right_input_batch, 64, dynamic=True)

    def _build_network(self, args):
        if args['correlation']:
            self._add_to_layers('conv1a', sharedLayers.conv2d(
                self._left_input_batch, [7, 7, 3, 64], strides=2, name='conv1'))
            self._add_to_layers('conv1b', sharedLayers.conv2d(self._right_input_batch, [
                                7, 7, 3, 64], strides=2, name='conv1', reuse=True))

            self._add_to_layers('conv2a', sharedLayers.conv2d(
                self._get_layer_as_input('conv1a'), [5, 5, 64, 128], strides=2, name='conv2'))
            self._add_to_layers('conv2b', sharedLayers.conv2d(self._get_layer_as_input(
                'conv1b'), [5, 5, 64, 128], strides=2, name='conv2', reuse=True))

            self._add_to_layers('conv_redir', sharedLayers.conv2d(self._get_layer_as_input(
                'conv2a'), [1, 1, 128, 64], strides=1, name='conv_redir'))
            self._add_to_layers('corr', sharedLayers.correlation(self._get_layer_as_input(
                'conv2a'), self._get_layer_as_input('conv2b'), max_disp=MAX_DISP))

            self._add_to_layers('conv3', sharedLayers.conv2d(tf.concat([self._get_layer_as_input('corr'), self._get_layer_as_input(
                'conv_redir')], axis=3), [5, 5, MAX_DISP * 2 + 1 + 64, 256], strides=2, name='conv3'))
        else:
            concat_inputs = tf.concat(
                [self._left_img_batch, self._right_input_batch], axis=-1)
            self._add_to_layers('conv1', sharedLayers.conv2d(
                concat_inputs, [7, 7, 6, 64], strides=2, name='conv1'))
            self._add_to_layers('conv2', sharedLayers.conv2d(
                self._get_layer_as_input('conv1'), [5, 5, 64, 128], strides=2, name='conv2'))
            self._add_to_layers('conv3', sharedLayers.conv2d(
                self._get_layer_as_input('conv2'), [5, 5, 128, 256], strides=2, name='conv3'))

        self._add_to_layers('conv3/1', sharedLayers.conv2d(
            self._get_layer_as_input('conv3'), [3, 3, 256, 256], strides=1, name='conv3/1'))
        self._add_to_layers('conv4', sharedLayers.conv2d(self._get_layer_as_input(
            'conv3/1'), [3, 3, 256, 512], strides=2, name='conv4'))
        self._add_to_layers('conv4/1', sharedLayers.conv2d(
            self._get_layer_as_input('conv4'), [3, 3, 512, 512], strides=1, name='conv4/1'))
        self._add_to_layers('conv5', sharedLayers.conv2d(self._get_layer_as_input(
            'conv4/1'), [3, 3, 512, 512], strides=2, name='conv5'))
        self._add_to_layers('conv5/1', sharedLayers.conv2d(
            self._get_layer_as_input('conv5'), [3, 3, 512, 512], strides=1, name='conv5/1'))
        self._add_to_layers('conv6', sharedLayers.conv2d(self._get_layer_as_input(
            'conv5/1'), [3, 3, 512, 1024], strides=2, name='conv6'))
        self._add_to_layers('conv6/1', sharedLayers.conv2d(self._get_layer_as_input(
            'conv6'), [3, 3, 1024, 1024], strides=1, name='conv6/1'))

        self._upsampling_block(self._get_layer_as_input(
            'conv6/1'), self._get_layer_as_input('conv5/1'), 1024, 512, 512, name='up5')

        self._upsampling_block(self._get_layer_as_input(
            'up5/concat'), self._get_layer_as_input('conv4/1'), 512, 256, 512, name='up4')

        self._upsampling_block(self._get_layer_as_input(
            'up4/concat'), self._get_layer_as_input('conv3/1'), 256, 128, 256, name='up3')

        if args['correlation']:
            self._upsampling_block(self._get_layer_as_input(
                'up3/concat'), self._get_layer_as_input('conv2a'), 128, 64, 128, name='up2')
        else:
            self._upsampling_block(self._get_layer_as_input(
                'up3/concat'), self._get_layer_as_input('conv2'), 128, 64, 128, name='up2')

        if args['correlation']:
            self._upsampling_block(self._get_layer_as_input(
                'up2/concat'), self._get_layer_as_input('conv1a'), 64, 32, 64, name='up1')
        else:
            self._upsampling_block(self._get_layer_as_input(
                'up2/concat'), self._get_layer_as_input('conv1'), 64, 32, 64, name='up1')

        self._add_to_layers('prediction', sharedLayers.conv2d(self._get_layer_as_input(
            'up1/concat'), [3, 3, 32, 1], strides=1, activation=lambda x: x, name='prediction'))
        self._disparities.append(self._make_disp(self._layers['prediction']))

         ############ LOOK BELOW IF DISPNET GIVES WRONG RESULTS #########################
        # rescaled_prediction = -preprocessing.rescale_image(self._layers['prediction'], tf.shape(self._left_input_batch)[1:3])
        ################################################################################
        rescaled_prediction = tf.image.resize_images(self._layers['prediction'], tf.shape(self._left_input_batch)[1:3]) * 2
        
        self._layers['rescaled_prediction'] = tf.image.resize_image_with_crop_or_pad(rescaled_prediction, self._restore_shape[0], self._restore_shape[1])
        self._disparities.append(self._layers['rescaled_prediction'])
