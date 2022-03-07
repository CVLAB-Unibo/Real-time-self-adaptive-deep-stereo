import tensorflow as tf
import numpy as np

def l1(x,y,mask=None):
	"""
	pixelwise reconstruction error
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return mask*tf.abs(x-y)

def l2(x,y,mask=None):
	"""
	PixelWise squarred error
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return mask*tf.square(x-y)

def mean_l1(x,y,mask=None):
	"""
	Mean reconstruction error
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return tf.reduce_sum(mask*tf.abs(x-y))/tf.reduce_sum(mask)

def mean_l2(x,y,mask=None):
	"""
	Mean squarred error
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return tf.reduce_sum(mask*tf.square(x-y))/tf.reduce_sum(mask)

def huber(x,y,c=1.0):
	diff = x-y
	l2 = tf.square(diff)
	l1 = tf.abs(diff)
	#c = (ratio)*tf.reduce_max(diff)
	diff = tf.where(tf.greater(diff,c),0.5*tf.square(c)+c*(l1-c),0.5*l2)
	return diff 

def mean_huber(x,y,mask=None):
	"""
	Mean huber loss
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	
	return tf.reduce_mean(huber(x,y)*mask)

def sum_huber(x,y,mask=None):
	"""
	Sum huber loss
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	
	return tf.reduce_sum(huber(x,y)*mask)

def sum_l1(x,y,mask=None):
	"""
	Sum of the reconstruction error
	Args:
		x: predicted image
		y: target image
		mask: compute only on this points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return tf.reduce_sum(mask*tf.abs(x-y))

def sum_l2(x,y,mask=None):
	"""
	Sum squarred error
	Args:
		x: predicted image
		y: target image
		mask: compute only on those points
	"""
	if mask is None:
		mask=tf.ones_like(x, dtype=tf.float32)
	return tf.reduce_sum(mask*tf.square(x-y))

def zncc(x,y):
	"""
	ZNCC dissimilarity measure
	Args:
		x: predicted image
		y: target image
	"""
	mean_x = tf.reduce_mean(x)
	mean_y = tf.reduce_mean(y)
	norm_x = x-mean_x
	norm_y = y-mean_y
	variance_x = tf.sqrt(tf.reduce_sum(tf.square(norm_x)))
	variance_y = tf.sqrt(tf.reduce_sum(tf.square(norm_y)))

	zncc = tf.reduce_sum(norm_x*norm_y)/(variance_x*variance_y)
	return 1-zncc


def SSIM(x, y):
	"""
	SSIM dissimilarity measure
	Args:
		x: predicted image
		y: target image
	"""
	C1 = 0.01**2
	C2 = 0.03**2
	mu_x = tf.nn.avg_pool(x,[1,3,3,1],[1,1,1,1],padding='VALID')
	mu_y = tf.nn.avg_pool(y,[1,3,3,1],[1,1,1,1],padding='VALID')
	
	sigma_x = tf.nn.avg_pool(x**2, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_x**2
	sigma_y = tf.nn.avg_pool(y**2, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_y**2
	sigma_xy = tf.nn.avg_pool(x*y, [1,3,3,1],[1,1,1,1],padding='VALID') - mu_x * mu_y

	SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
	SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

	SSIM = SSIM_n / SSIM_d

	return tf.clip_by_value((1-SSIM)/2, 0 ,1)

def ssim_l1(x,y,alpha=0.85):
	ss = tf.pad(SSIM(x,y),[[0,0],[1,1],[1,1],[0,0]])
	ll = l1(x,y)
	return alpha*ss+(1-alpha)*ll

def mean_SSIM(x,y):
	"""
	Mean error over SSIM reconstruction
	"""
	return tf.reduce_mean(SSIM(x,y))


def mean_SSIM_L1(x, y):
	return 0.85* mean_SSIM(x, y) + 0.15 * mean_l1(x, y)


def sign_and_elementwise(x,y):
	"""
	Return the elementwise and of the sign between vectors
	"""
	element_wise_sign = tf.sigmoid(10*(tf.sign(x)*tf.sign(y)))
	return tf.reduce_mean(tf.sigmoid(element_wise_sign))

def cos_similarity(x,y,normalize=False):
	"""
	Return the cosine similarity between (normalized) vectors
	"""
	if normalize:
		x = tf.nn.l2_normalize(x)
		y = tf.nn.l2_normalize(y)
	return tf.reduce_sum(x*y)

def smoothness(x, y):
	"""
	Smoothness constraint between predicted and image 
	Args:
		x: disparity
		y: image
	"""
	def gradient_x(image):
		sobel_x = tf.Variable(initial_value=[[1,0,-1],[2,0,-2],[1,0,-1]],trainable=False,dtype=tf.float32)
		sobel_x = tf.reshape(sobel_x,[3,3,1,1])
		if image.get_shape()[-1].value==3:
			sobel_x = tf.concat([sobel_x,sobel_x,sobel_x],axis=2)
		return tf.nn.conv2d(image,sobel_x,[1,1,1,1],padding='SAME')
	
	def gradient_y(image):
		sobel_y = tf.Variable(initial_value=[[1,2,-1],[0,0,0],[-1,-2,-1]],trainable=False,dtype=tf.float32)
		sobel_y = tf.reshape(sobel_y,[3,3,1,1])
		if image.get_shape()[-1].value==3:
			sobel_y = tf.concat([sobel_y,sobel_y,sobel_y],axis=2)
		return tf.nn.conv2d(image,sobel_y,[1,1,1,1],padding='SAME')
	
	#normalize image and disp in a fixed range
	x = x/255
	y = y/255

	disp_gradients_x = gradient_x(x)
	disp_gradients_y = gradient_y(x)

	image_gradients_x = tf.reduce_mean(gradient_x(y), axis=-1, keepdims=True) 
	image_gradients_y = tf.reduce_mean(gradient_y(y), axis=-1, keepdims=True)

	weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True)) 
	weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

	smoothness_x = tf.abs(disp_gradients_x) * weights_x
	smoothness_y = tf.abs(disp_gradients_y) * weights_y

	return tf.reduce_mean(smoothness_x + smoothness_y)





###################################################################################################################################################################

from Data_utils import preprocessing

SUPERVISED_LOSS={
	'mean_l1':mean_l1,
	'sum_l1':sum_l1,
	'mean_l2':mean_l2,
	'sum_l2':sum_l2,
	'mean_SSIM':mean_SSIM,
	'mean_SSIM_l1':mean_SSIM_L1,
	'ZNCC':zncc,
	'cos_similarity':cos_similarity,
	'smoothness':smoothness,
	'mean_huber':mean_huber,
	'sum_huber':sum_huber
}

PIXELWISE_LOSSES={
	'l1':l1,
	'l2':l2,
	'SSIM':SSIM,
	'huber':huber,
	'ssim_l1':ssim_l1
}

ALL_LOSSES = dict(SUPERVISED_LOSS)
ALL_LOSSES.update(PIXELWISE_LOSSES)


def get_supervised_loss(name, multiScale=False, logs=False, weights=None, reduced=True, max_disp=None):
	"""
	Build a lambda op to be used to compute a loss function
	Args:
		name: name of the loss function to build
		multiScale: if True compute multiple loss, one for each scale at which disparities are predicted
		logs: if True enable tf summary
		weights: array of weights to be multiplied for the losses at different resolution
		reduced: if true return the sum of the loss across the different scales, false to return an array with the different losses
		max_disp: if different from None clip max disparity to be this one
	"""
	if name not in ALL_LOSSES.keys():
		print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
		raise Exception('Unknown loss function selected')
	
	base_loss_function = ALL_LOSSES[name]
	if weights is None:
		weights = [1]*10
	if max_disp is None:
		max_disp=1000
	def compute_loss(disparities,inputs):
		left = inputs['left']
		right = inputs['right']
		targets = inputs['target']
		accumulator=[]
		if multiScale:
			disp_to_test=len(disparities)
		else:
			disp_to_test=1
		valid_map = tf.where(tf.logical_or(tf.equal(targets, 0), tf.greater_equal(targets,max_disp)), tf.zeros_like(targets, dtype=tf.float32), tf.ones_like(targets, dtype=tf.float32))
		
		for i in range(0,disp_to_test):
			#upsample prediction
			current_disp = disparities[-(i+1)]
			disparity_scale_factor = tf.cast(tf.shape(left)[2],tf.float32)/tf.cast(tf.shape(current_disp)[2],tf.float32)
			resized_disp = preprocessing.resize_to_prediction(current_disp,targets) * disparity_scale_factor

			partial_loss = base_loss_function(resized_disp,targets,valid_map)
			#partial_loss = tf.Print(partial_loss,[disparity_scale_factor,tf.shape(valid_map),tf.reduce_sum(valid_map), tf.reduce_sum(valid_map*resized_disp)/tf.reduce_sum(valid_map), tf.reduce_sum(valid_map*targets)/tf.reduce_sum(valid_map)],summarize=10000)
			if logs:
				tf.summary.scalar('Loss_resolution_{}'.format(i),partial_loss)
			accumulator.append(weights[i]*partial_loss)
		if reduced:
			return tf.reduce_sum(accumulator)
		else:
			return accumulator
	return compute_loss

def get_proxy_loss(name, multiScale=False, logs=False, weights=None, reduced=True, max_disp=None):
	"""
	Build a lambda op to be used to compute a loss function
	Args:
		name: name of the loss function to build
		multiScale: if True compute multiple loss, one for each scale at which disparities are predicted
		logs: if True enable tf summary
		weights: array of weights to be multiplied for the losses at different resolution
		reduced: if true return the sum of the loss across the different scales, false to return an array with the different losses
		max_disp: if different from None clip max disparity to be this one
	"""
	if name not in ALL_LOSSES.keys():
		print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
		raise Exception('Unknown loss function selected')
	
	base_loss_function = ALL_LOSSES[name]
	if weights is None:
		weights = [0.01]*10
	if max_disp is None:
		max_disp=1000
	def compute_loss(disparities,inputs):
		left = inputs['left']
		right = inputs['right']
		targets = inputs['target']
		proxies = inputs['proxy']
		accumulator=[]
		if multiScale:
			disp_to_test=len(disparities)
		else:
			disp_to_test=1
		valid_map = tf.where(tf.logical_or(tf.less_equal(proxies, 0), tf.greater_equal(proxies,192)), tf.zeros_like(targets, dtype=tf.float32), tf.ones_like(targets, dtype=tf.float32))

		for i in range(0,disp_to_test):
			#upsample prediction
			current_disp = disparities[-(i+1)]
			disparity_scale_factor = tf.cast(tf.shape(left)[2],tf.float32)/tf.cast(tf.shape(current_disp)[2],tf.float32)
			resized_disp = preprocessing.resize_to_prediction(current_disp,targets) * disparity_scale_factor

			partial_loss = base_loss_function(resized_disp,proxies,valid_map)
			#partial_loss = tf.Print(partial_loss,[disparity_scale_factor,tf.shape(valid_map),tf.reduce_sum(valid_map), tf.reduce_sum(valid_map*resized_disp)/tf.reduce_sum(valid_map), tf.reduce_sum(valid_map*targets)/tf.reduce_sum(valid_map)],summarize=10000)
			if logs:
				tf.summary.scalar('Loss_resolution_{}'.format(i),partial_loss)
			accumulator.append(weights[i]*partial_loss)
		if reduced:
			return tf.reduce_sum(accumulator)
		else:
			return accumulator
	return compute_loss

def get_reprojection_loss(reconstruction_loss,multiScale=False, logs=False, weights=None,reduced=True):
	"""
	Build a lmbda op to be used to compute a loss function using reprojection between left and right frame
	Args:
		reconstruction_loss: name of the loss function used to compare reprojected and real image
		multiScale: if True compute multiple loss, one for each scale at which disparities are predicted
		logs: if True enable tf summary
		weights: array of weights to be multiplied for the losses at different resolution
		reduced: if true return the sum of the loss across the different scales, false to return an array with the different losses
	"""
	if reconstruction_loss not in ALL_LOSSES.keys():
		print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
		raise Exception('Unknown loss function selected')
	base_loss_function = ALL_LOSSES[reconstruction_loss]
	if weights is None:
		weights = [1]*10
	def compute_loss(disparities,inputs):
		left = inputs['left']
		right = inputs['right']
		#normalize image to be between 0 and 1 
		left = tf.cast(left,dtype=tf.float32)/256.0
		right = tf.cast(right,dtype=tf.float32)/256.0
		accumulator=[]
		if multiScale:
			disp_to_test=len(disparities)
		else:
			disp_to_test=1
		for i in range(disp_to_test):
			#rescale prediction to full resolution
			current_disp = disparities[-(i+1)]
			disparity_scale_factor = tf.cast(tf.shape(left)[2],tf.float32)/tf.cast(tf.shape(current_disp)[2],tf.float32)
			resized_disp = preprocessing.resize_to_prediction(current_disp, left) * disparity_scale_factor

			reprojected_left = preprocessing.warp_image(right, resized_disp)
			partial_loss = base_loss_function(reprojected_left,left)
			if logs:
				tf.summary.scalar('Loss_resolution_{}'.format(i),partial_loss)
			accumulator.append(weights[i]*partial_loss)
		if reduced:
			return tf.reduce_sum(accumulator)
		else:
			return accumulator
	return compute_loss
