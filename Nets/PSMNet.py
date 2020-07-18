import tensorflow as tf
import tensorflow.keras as keras
from .utils import *
import numpy as np

from Nets import Stereo_net
from Nets import sharedLayers
from Data_utils import preprocessing

class PSMNet(Stereo_net.StereoNet):
    _valid_args = [
        ("left_img", "meta op for left image batch"),
        ("right_img", "meta op for right image batch"),
        ("max_disp", "max_disp for the model")
    ] + Stereo_net.StereoNet._valid_args
    _netName = "PSMNet"

    def __init__(self, **kwargs):
        """
        Creation of a PSMNet for stereo prediction
        """
        super(PSMNet, self).__init__(**kwargs)

    def _validate_args(self, args):
        """
        Check that args contains everything that is needed
        Valid Keys for args:
            left_img: left image op
            right_img: right image op
        """
        super(PSMNet, self)._validate_args(args)
        if ("left_img" not in args) or ("right_img" not in args):
            raise Exception('Missing input op for left and right images')
        if 'max_disp' not in args:
            print('WARNING: max_disp not setted, setting default value 192')
            args['max_disp'] = 192
        return args

    def _preprocess_inputs(self, args):
        self._left_input_batch = args['left_img']
        self._restore_shape = tf.shape(args['left_img'])[1:3]
        self._left_input_batch = tf.cast(self._left_input_batch, tf.float32)
        self._left_input_batch = preprocessing.pad_image(
            self._left_input_batch, 64)

        self._right_input_batch = args['right_img']
        self._right_input_batch = tf.cast(self._right_input_batch, tf.float32)
        self._right_input_batch = preprocessing.pad_image(
            self._right_input_batch, 64)
        self.max_disp = args['max_disp']
        self.image_size_tf = tf.shape(self._left_input_batch)[1:3]

    def _build_network(self, args):
        conv4_left = self.CNN(self._left_input_batch)
        fusion_left = self.SPP(conv4_left)

        conv4_right = self.CNN(self._right_input_batch, reuse=tf.compat.v1.AUTO_REUSE)
        fusion_right = self.SPP(conv4_right, reuse=tf.compat.v1.AUTO_REUSE)

        cost_vol = self.cost_vol(fusion_left, fusion_right, self.max_disp)
        outputs = self.CNN3D(cost_vol)
        disps = self.output(outputs)#size of (B, H, W),3out
        self._disparities.append(disps[2])

    def CNN(self, bottom, reuse=False):
        with tf.compat.v1.variable_scope('CNN',reuse=reuse):
            with tf.compat.v1.variable_scope('conv0',reuse=reuse):
                if reuse == tf.compat.v1.AUTO_REUSE:
                    tf.print(bottom, ['left_batch_size:',bottom])
                self._add_to_layers('CNN/conv0/conv0_1', sharedLayers.conv2d(bottom, [3, 3, 3, 32], \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,strides=2, name='conv0_1',reuse=reuse))
                for i in range(1, 3):
                    bottom = self._get_layer_as_input('CNN/conv0/conv0_'+str(i))
                    self._add_to_layers('CNN/conv0/conv0_'+str(i+1), sharedLayers.conv2d(bottom,\
                         [3, 3, 32, 32], activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True, strides=1, name='conv0_'+str(i+1),reuse=reuse))
            with tf.compat.v1.variable_scope('conv1',reuse=reuse):
                bottom = self._get_layer_as_input('CNN/conv0/conv0_3')
                for i in range(3):
                    bottom = self.res_block(bottom, 32, 3, layer_prefix='CNN/conv1/conv1_%d' % (i+1), reuse=reuse)
            with tf.compat.v1.variable_scope('conv2',reuse=reuse):
                bottom = self.res_block(bottom, 64, 3, strides=2, layer_prefix='CNN/conv2/conv2_1', reuse=reuse, projection=True)
                for i in range(1, 16):
                    bottom = self.res_block(bottom, 64, 3, layer_prefix='CNN/conv2/conv2_%d' % (i+1), reuse=reuse)
            with tf.compat.v1.variable_scope('conv3',reuse=reuse):
                bottom = self.res_block(bottom, 128, 3, dilation_rate=1, layer_prefix='CNN/conv3/conv3_1', reuse=reuse,projection=True)
                for i in range(1, 3):
                    bottom = self.res_block(bottom, 128, 3, dilation_rate=1, layer_prefix='CNN/conv3/conv3_%d' % (i+1), reuse=reuse)
            with tf.compat.v1.variable_scope('conv4',reuse=reuse):
                for i in range(3):
                    bottom = self.res_block(bottom, 128, 3, dilation_rate=2, layer_prefix='CNN/conv4/conv4_%d' % (i+1), reuse=reuse)
        return bottom

    def res_block(self, bottom, filters, kernel_size, strides=1, dilation_rate=1, layer_prefix=None, reuse=None, projection=False):
        with tf.compat.v1.variable_scope(layer_prefix,reuse=reuse):
            names = []
            names.append('{}/conv1'.format(layer_prefix))

            short_cut = bottom
            input_layer = bottom
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [kernel_size, kernel_size, input_layer.get_shape()[-1].value, filters], \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,strides=strides, name='conv1', dilation_rate=dilation_rate, reuse=reuse))
            
            names.append('{}/conv2'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv2d(input_layer, [kernel_size, kernel_size, input_layer.get_shape()[-1].value, filters], \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False,strides=1, name='conv2', dilation_rate=dilation_rate, reuse=reuse))
            
            if projection:
                names.append('{}/projection'.format(layer_prefix))
                self._add_to_layers(names[-1], sharedLayers.conv2d(short_cut, [1, 1, short_cut.get_shape()[-1].value, filters], \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True, strides=strides, name='projection', reuse=reuse))
                short_cut = self._get_layer_as_input(names[-1])
            
            if projection:
                input_layer = self._get_layer_as_input(names[-2])
            else:
                input_layer = self._get_layer_as_input(names[-1])
            bottom = tf.add(input_layer, short_cut, 'add')
            names.append('{}/leaky_relu'.format(layer_prefix))
            self._add_to_layers(names[-1],tf.nn.leaky_relu(bottom, name='leaky_relu'))
        return self._get_layer_as_input(names[-1])

    def SPP(self, bottom, reuse=False):
        with tf.compat.v1.variable_scope('SPP',reuse=reuse):
            branches = []
            for i, p in enumerate([64, 32, 16, 8]):
                branches.append(self.SPP_branch(bottom, p, 32, 3, name='branch_%d' % (i+1), reuse=reuse))
            #if not reuse:                
            #conv2_16 = tf.compat.v1.get_default_graph().get_tensor_by_name('CNN/conv2/conv2_16/add:0')
            #conv4_3 = tf.compat.v1.get_default_graph().get_tensor_by_name('CNN/conv4/conv4_3/add:0')
            conv2_16 = self._get_layer_as_input('CNN/conv2/conv2_16/leaky_relu')
            conv4_3 = self._get_layer_as_input('CNN/conv4/conv4_3/leaky_relu')
            #else:
            #    conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv2/conv2_16/add:0')
            #    conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv4/conv4_3/add:0')            
            concat = tf.concat([conv2_16, conv4_3] + branches, axis=-1, name='concat')
            with tf.compat.v1.variable_scope('fusion',reuse=reuse):
                self._add_to_layers('SPP/fusion/conv1', sharedLayers.conv2d(concat, [3, 3, concat.get_shape()[-1].value, 128], name='conv1', \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,reuse=reuse))
                input_layer = self._get_layer_as_input('SPP/fusion/conv1')
                self._add_to_layers('SPP/fusion/conv2', sharedLayers.conv2d(input_layer,[1,1,input_layer.get_shape()[-1].value,32], name='conv2', \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,reuse=reuse))
        return self._get_layer_as_input('SPP/fusion/conv2')

    def SPP_branch(self, bottom, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, apply_bn=True, apply_relu=True):
        with tf.compat.v1.variable_scope(name,reuse=reuse):
            size = tf.shape(bottom)[1:3]
            ap = keras.layers.AveragePooling2D(pool_size=(pool_size,pool_size), padding='same', name='avg_pool')
            bottom = ap(bottom)
            
            self._add_to_layers('SPP/'+ name + '/conv', sharedLayers.conv2d(bottom,[kernel_size, kernel_size, bottom.get_shape()[-1].value, filters], \
                strides=strides, name='conv', reuse=reuse, activation=tf.nn.leaky_relu, batch_norm=apply_bn, apply_relu=apply_relu))
            bottom = tf.image.resize(self._get_layer_as_input('SPP/'+ name + '/conv'), size)
        return bottom

    def cost_vol(self, left, right, max_disp=192):
        with tf.compat.v1.variable_scope('cost_vol'):
            disparity_costs = []
            #shape = tf.shape(right) #(N,H,W,F)
            for i in range(max_disp // 4):
                if i > 0:
                    left_tensor = keras.backend.spatial_2d_padding(left[:, :, i:, :], padding=((0, 0), (i, 0)))
                    right_tensor = keras.backend.spatial_2d_padding(right[:, :, :-i, :], padding=((0, 0), (i, 0)))
                    cost = tf.concat([left_tensor, right_tensor], axis=3)
                else:
                    cost = tf.concat([left, right], axis=3)
                disparity_costs.append(cost)
            
            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol
    
    def CNN3D(self, bottom):
        with tf.compat.v1.variable_scope('CNN3D'):
            for i in range(2):
                self._add_to_layers('CNN3D/3Dconv0_%d' % (i+1), sharedLayers.conv3d(bottom, [3, 3, 3, bottom.get_shape()[-1].value,32], \
                    activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True, name='3Dconv0_%d' % (i+1)))

            bottom = self._get_layer_as_input('CNN3D/3Dconv0_2')
            _3Dconv1 = self.res_block_3d(bottom, 32, 3, layer_prefix='CNN3D/3Dconv1')

            _3Dstack = [self.hourglass('3d', _3Dconv1, [64, 64, 64, 32], [3, 3, 3, 3], [None, None, -2, _3Dconv1],name='3Dstack1')]
            for i in range(1, 3):
                _3Dstack.append(self.hourglass('3d', _3Dstack[-1][-1], [64, 64, 64, 32], [3, 3, 3, 3],
                                          [_3Dstack[-1][-2], None, _3Dstack[0][0], _3Dconv1], name='3Dstack%d' % (i+1)))

            self._add_to_layers('output_1_1', sharedLayers.conv3d(_3Dstack[0][3], [3,3,3,_3Dstack[0][3].get_shape()[-1].value,32], name='output_1_1'))
            input_layer = self._get_layer_as_input('output_1_1')
            self._add_to_layers('output_1', sharedLayers.conv3d(input_layer, [3,3,3,input_layer.get_shape()[-1].value,1],name='output_1', batch_norm=False, apply_relu=False))
            
            output_1 = self._get_layer_as_input('output_1')
            outputs = [output_1]

            for i in range(1, 3):
                self._add_to_layers('output_%d_1' % (i+1), sharedLayers.conv3d(_3Dstack[i][3],[3,3,3,_3Dstack[i][3].get_shape()[-1].value,32], \
                    name='output_%d_1' % (i+1)))
                input_layer = self._get_layer_as_input('output_%d_1' % (i+1))
                self._add_to_layers('output_%d_2' % (i+1), sharedLayers.conv3d(input_layer,[3,3,3,input_layer.get_shape()[-1].value,1], 
                name='output_%d_2' % (i+1), batch_norm=False, apply_relu=False))
                prev = self._get_layer_as_input('output_%d_2' % (i+1))
                output = tf.add(prev, outputs[-1], name='output_%d' % (i+1))
                outputs.append(output)

        return outputs

    def res_block_3d(self, bottom, filters, kernel_size, strides=1, dilation_rate=1, layer_prefix=None, reuse=None, projection=False):
        with tf.compat.v1.variable_scope(layer_prefix,reuse=reuse):
            names = []
            names.append('{}/conv1'.format(layer_prefix))

            short_cut = bottom
            input_layer = bottom
            self._add_to_layers(names[-1], sharedLayers.conv3d(input_layer, [kernel_size, kernel_size, kernel_size, input_layer.get_shape()[-1].value, filters], \
                    strides=strides, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,name='conv1', reuse=reuse))
            
            names.append('{}/conv2'.format(layer_prefix))
            input_layer = self._get_layer_as_input(names[-2])
            self._add_to_layers(names[-1], sharedLayers.conv3d(input_layer, [kernel_size, kernel_size, kernel_size, input_layer.get_shape()[-1].value, filters], \
                    strides=1, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False,name='conv2', reuse=reuse))
            
            if projection:
                names.append('{}/projection'.format(layer_prefix))
                input_layer = short_cut
                self._add_to_layers(names[-1], sharedLayers.conv3d(input_layer, [1, 1, 1, input_layer.get_shape()[-1].value, filters], \
                    strides=strides, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True,name='projection', reuse=reuse))
            
            if projection:
                input_layer = self._get_layer_as_input(names[-2])
            else:
                input_layer = self._get_layer_as_input(names[-1])
            bottom = tf.add(input_layer, short_cut, 'add')
            bottom = tf.nn.relu(bottom, name='relu')
        return bottom

    def hourglass(self,strs, bottom, filters_list, kernel_size_list, short_cut_list, dilation_rate=1, name=None):
        with tf.compat.v1.variable_scope(name):
            output = []
            
            for i, (filters, kernel_size, short_cut) in enumerate(zip(filters_list, kernel_size_list, short_cut_list)):
                if i < len(filters_list) // 2:
                    if strs == '2d':
                        self._add_to_layers(name + '/stack_%d_1' % (i+1), sharedLayers.conv2d(bottom,[kernel_size, kernel_size, \
                            bottom.get_shape()[-1].value, filters], strides=2, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True, name='/stack_%d_1' % (i+1)))
                        bottom = self._get_layer_as_input(name+'/stack_%d_1'%(i+1))
                        self._add_to_layers(name + '/stack_%d_2' % (i+1), sharedLayers.conv2d(bottom,[kernel_size, kernel_size, \
                        bottom.get_shape()[-1].value, filters], strides=1, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False, name='/stack_%d_2' % (i+1)))
                    elif strs == '3d':
                        self._add_to_layers(name + '/stack_%d_1' % (i+1), sharedLayers.conv3d(bottom,[kernel_size, kernel_size, kernel_size, \
                        bottom.get_shape()[-1].value, filters], strides=2, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=True, name='/stack_%d_1' % (i+1)))
                        bottom = self._get_layer_as_input(name+'/stack_%d_1'%(i+1))
                        self._add_to_layers(name + '/stack_%d_2' % (i+1), sharedLayers.conv3d(bottom,[kernel_size, kernel_size, kernel_size, \
                        bottom.get_shape()[-1].value, filters], strides=1, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False, name='/stack_%d_2' % (i+1)))

                    if short_cut is not None:
                        if type(short_cut) is int:
                            short_cut = output[short_cut]
                        bottom = tf.add(bottom, short_cut, name='stack_%d' % (i+1))
                    bottom = tf.nn.leaky_relu(bottom, name='relu')
                else:
                    #反卷积有问题,必须确定batch-size,height,weight才能正确运行
                    #第三层相加之后在执行relu，第四层卷积运算完之后不执行relu直接相加
                    if strs == '2d':
                        self._add_to_layers(name + '/stack_%d_1' % (i+1), sharedLayers.conv2d_transpose(bottom,[kernel_size, kernel_size, \
                            filters,bottom.get_shape()[-1].value ], strides=2, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False, name='/stack_%d_1' % (i+1)))
                    elif strs == '3d':
                        self._add_to_layers(name + '/stack_%d_1' % (i+1), sharedLayers.conv3d_transpose(bottom, [kernel_size, kernel_size, kernel_size, filters,\
                            bottom.get_shape()[-1].value], strides=2, activation=tf.nn.leaky_relu, batch_norm=True, apply_relu=False, name='/stack_%d_1' % (i+1)))
                    
                    bottom = self._get_layer_as_input(name + '/stack_%d_1' % (i+1))
                    if short_cut is not None:
                        if type(short_cut) is int:
                            short_cut = output[short_cut]
                        bottom = tf.add(bottom, short_cut, name='stack_%d' % (i + 1))
                    if i == 2:
                        bottom = tf.nn.leaky_relu(bottom, name='relu')
                output.append(bottom)
        return output

    def output(self, outputs):
        disps = []
        for i, output in enumerate(outputs):
            squeeze = tf.squeeze(output, [4])
            transpose = tf.transpose(squeeze, [0, 2, 3, 1])
            upsample = tf.transpose(tf.image.resize(transpose, self.image_size_tf), [0, 3, 1, 2])
            disps.append(self.soft_arg_min(upsample, 'soft_arg_min_%d' % (i+1)))
        return disps

    def soft_arg_min(self, filtered_cost_volume, name):
        with tf.compat.v1.variable_scope(name):
            #input.shape (batch, depth, H, W)
            # softargmin to disp image, outsize of (B, H, W)

            #print('filtered_cost_volume:',filtered_cost_volume.shape)
            probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                            axis=1, name='prob_volume')
            #print('probability_volume:',probability_volume.shape)
            volume_shape = tf.shape(probability_volume)
            soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
            soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))
            soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
            soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
            estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)
            #print(estimated_disp_image.shape)
            estimated_disp_image = tf.expand_dims(estimated_disp_image, axis=3)
            return estimated_disp_image
