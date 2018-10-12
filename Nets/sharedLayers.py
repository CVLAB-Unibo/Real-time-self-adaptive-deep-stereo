import tensorflow as tf
import os

INITIALIZER_CONV = tf.contrib.layers.xavier_initializer()
INITIALIZER_BIAS = tf.constant_initializer(0.0)
MODE = 'TF'

####################################################
#### Uncomment lines below for cuda correlation ####
####################################################

#REPO_DIR = os.path.dirname(os.path.abspath(__file__))
#shift_corr_module = tf.load_op_library(os.path.join(REPO_DIR, 'Native/shift_corr.so'))

#@tf.RegisterGradient("ShiftCorr")
#def _ShiftCorrOpGrad(op, grad):
#	return shift_corr_module.shift_corr_grad(op.inputs[0], op.inputs[1], grad, max_disp=op.get_attr('max_disp'))

# MODE='CUDA'

#######################################################

def correlation(x,y,max_disp, name='corr', mode=MODE,stride=1):
	if mode == 'TF':
		return correlation_tf(x,y,max_disp,name=name,stride=stride)
	else:
		if stride!=1:
			raise Exception('Cuda implementation cannot handle stride different than 1')
		return correlation_native(x,y,max_disp,name=name)

def correlation_native(x, y, max_disp, name='corr'):
	with tf.variable_scope(name):
		input_shape = x.get_shape().as_list()
		x = tf.pad(x, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
		y = tf.pad(y, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]], "CONSTANT")
		corr = shift_corr_module.shift_corr(x, y, max_disp=max_disp)
		corr = tf.transpose(corr, perm=[0, 2, 3, 1])
		corr.set_shape([input_shape[0],input_shape[1],input_shape[2],2*max_disp+1])
		return corr

def correlation_tf(x, y, max_disp, stride=1, name='corr'):
	with tf.variable_scope(name):
		corr_tensors = []
		y_shape = tf.shape(y)
		y_feature = tf.pad(y,[[0,0],[0,0],[max_disp,max_disp],[0,0]])
		for i in range(-max_disp, max_disp+1,stride):
			shifted = tf.slice(y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1])
			corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))

		result = tf.concat(corr_tensors,axis=-1)
		return result


def conv2d(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS)
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x


def dilated_conv2d(x, kernel_shape, rate=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='dilated_conv', reuse=False, wName='weights', bName='biases',  batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(
            wName, kernel_shape, initializer=INITIALIZER_CONV)
        biases = tf.get_variable(
            bName, kernel_shape[3], initializer=INITIALIZER_BIAS)
        x = tf.nn.atrous_conv2d(x, weights, rate=rate, padding=padding)
        x = tf.nn.bias_add(x, biases)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x


def conv2d_transpose(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), name='conv', reuse=False, wName='weights', bName='bias',  batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(wName, kernel_shape,initializer=INITIALIZER_CONV)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
        b = tf.get_variable(bName, kernel_shape[2], initializer=INITIALIZER_BIAS)
        x_shape = tf.shape(x)
        output_shape = [x_shape[0], x_shape[1] * strides,x_shape[2] * strides, kernel_shape[2]]
        x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)
        return x

def depthwise_conv(x, kernel_shape, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=False, training=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(wName, kernel_shape, initializer=INITIALIZER_CONV)
        b = tf.get_variable(bName, kernel_shape[3]*kernel_shape[2], initializer=INITIALIZER_BIAS)
        x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=training,momentum=0.99)
        x = activation(x)
        return x

def separable_conv2d(x, kernel_shape, channel_multiplier=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False):
    with tf.variable_scope(name, reuse=reuse):
        #detpthwise conv
        depthwise_conv_kernel = [kernel_shape[0],kernel_shape[1],kernel_shape[2],channel_multiplier]
        x = depthwise_conv(x,depthwise_conv_kernel,strides=strides,activation=lambda x: tf.maximum(0.1 * x, x),padding=padding,name='depthwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training)

        #pointwise_conv
        pointwise_conv_kernel = [1,1,x.get_shape()[-1].value,kernel_shape[-1]]
        x = conv2d(x,pointwise_conv_kernel,strides=strides,activation=activation,padding=padding,name='pointwise_conv',reuse=reuse,wName=wName,bName=bName,batch_norm=batch_norm, training=training)

        return x

def grouped_conv2d(x, kernel_shape, num_groups=1, strides=1, activation=lambda x: tf.maximum(0.1 * x, x), padding='SAME', name='conv', reuse=False, wName='weights', bName='bias', batch_norm=True, training=False):
    with tf.variable_scope(name,reuse=reuse):
        w = tf.get_variable(wName,shape=kernel_shape,initializer=INITIALIZER_CONV)
        b = tf.get_variable(bName, kernel_shape[3], initializer=INITIALIZER_BIAS)

        input_groups = tf.split(x,num_or_size_splits=num_groups,axis=-1)
        kernel_groups = tf.split(w, num_or_size_splits=num_groups, axis=2)
        bias_group = tf.split(b,num_or_size_splits=num_groups,axis=-1)
        output_groups = [tf.nn.conv2d(i, k,[1,strides,strides,1],padding=padding)+bb for i, k,bb in zip(input_groups, kernel_groups,bias_group)]
		# Concatenate the groups
        x = tf.concat(output_groups,axis=3)
        if batch_norm:
            x = tf.layers.batch_normalization(x,training=training,momentum=0.99)
        x = activation(x)  
        return x

def channel_shuffle_inside_group(x, num_groups, name='shuffle'):
    with tf.variable_scope(name):
        _, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
    return output
