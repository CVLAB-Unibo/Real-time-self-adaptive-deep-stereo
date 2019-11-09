import tensorflow as tf
from matplotlib import cm
import numpy as np

FULLY_DIFFERENTIABLE=False

def pad_image(immy,down_factor = 256,dynamic=False):
    """
    pad image with a proper number of 0 to prevent problem when concatenating after upconv
    Args:
        immy: metaop that produces an image
        down_factor: downgrade resolution that should be respected before feeding the image to the network
        dynamic: if dynamic is True use dynamic shape of immy, otherway use static shape
    """
    if dynamic:
        immy_shape = tf.shape(immy)
        new_height = tf.where(tf.equal(immy_shape[-3]%down_factor,0),x=immy_shape[-3],y=(tf.floordiv(immy_shape[-3],down_factor)+1)*down_factor)
        new_width = tf.where(tf.equal(immy_shape[-2]%down_factor,0),x=immy_shape[-2],y=(tf.floordiv(immy_shape[-2],down_factor)+1)*down_factor)
    else:
        immy_shape = immy.get_shape().as_list()
        new_height = immy_shape[-3] if immy_shape[-3]%down_factor==0 else ((immy_shape[-3]//down_factor)+1)*down_factor
        new_width = immy_shape[-2] if immy_shape[-2]%down_factor==0 else ((immy_shape[-2]//down_factor)+1)*down_factor
    
    pad_height_left = (new_height-immy_shape[-3])//2
    pad_height_right = (new_height-immy_shape[-3]+1)//2
    pad_width_left = (new_width-immy_shape[-2])//2
    pad_width_right = (new_width-immy_shape[-2]+1)//2
    immy = tf.pad(immy,[[0,0],[pad_height_left,pad_height_right],[pad_width_left,pad_width_right],[0,0]],mode="REFLECT")
    return immy

def random_crop(crop_shape, tensor_list):
	"""
	Perform an alligned random crop on the list of tensors passed as arguments l r and gt
	"""
	image_shape = tf.shape(tensor_list[0])
	max_row = image_shape[0]-crop_shape[0]-1
	max_col = image_shape[1]-crop_shape[1]-1
	max_row = tf.cond(max_row>0, lambda: max_row, lambda: 1)
	max_col = tf.cond(max_col>0, lambda: max_col, lambda: 1)
	start_row = tf.random_uniform([],minval=0,maxval=max_row,dtype=tf.int32)
	start_col = tf.random_uniform([],minval=0,maxval=max_col,dtype=tf.int32)
	result=[]
	for x in tensor_list:
		static_shape = x.get_shape().as_list()
		if len(static_shape)==3:
			#crop
			temp = x[start_row:start_row+crop_shape[0],start_col:start_col+crop_shape[1],:]
			#force shape
			temp.set_shape([crop_shape[0],crop_shape[1],static_shape[-1]])
		else:
			#crop
			temp = x[:,start_row:start_row+crop_shape[0],start_col:start_col+crop_shape[1],:]
			#force shape
			temp.set_shape([static_shape[0],crop_shape[0],crop_shape[1],static_shape[-1]])
		result.append(temp)
	return result

	


def augment(left_img, right_img):
    active = tf.random_uniform(shape=[4], minval=0, maxval=1, dtype=tf.float32)
    left_img = tf.cast(left_img,tf.float32)
    right_img = tf.cast(right_img,tf.float32)

    # random gamma
    # random_gamma = tf.random_uniform(shape=(),minval=0.95,maxval=1.05,dtype=tf.float32)
    # left_img = tf.where(active[0]>0.5,left_img,tf.image.adjust_gamma(left_img,random_gamma))
    # right_img = tf.where(active[0]>0.5,right_img,tf.image.adjust_gamma(right_img,random_gamma))

    # random brightness
    random_delta = tf.random_uniform(shape=(), minval=-0.05, maxval=0.05, dtype=tf.float32)
    left_img = tf.where(active[1] > 0.5, left_img, tf.image.adjust_brightness(left_img, random_delta))
    right_img = tf.where(active[1] > 0.5, right_img, tf.image.adjust_brightness(right_img, random_delta))

    # random contrast
    random_contrast = tf.random_uniform(shape=(), minval=0.8, maxval=1.2, dtype=tf.float32)
    left_img = tf.where(active[2] > 0.5, left_img, tf.image.adjust_contrast(left_img, random_contrast))
    right_img = tf.where(active[2] > 0.5, right_img, tf.image.adjust_contrast(right_img, random_contrast))

    # random hue
    random_hue = tf.random_uniform(shape=(), minval=0.8, maxval=1.2, dtype=tf.float32)
    left_img = tf.where(active[3] > 0.5, left_img,tf.image.adjust_hue(left_img, random_hue))
    right_img = tf.where(active[3] > 0.5, right_img, tf.image.adjust_hue(right_img, random_hue))

    left_img = tf.clip_by_value(left_img,0,255)
    right_img = tf.clip_by_value(right_img,0,255)

    return left_img,right_img

def colorize_img(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping to a grayscale colormap.
    Arguments:
      - value: 4D Tensor of shape [batch_size,height, width,1]
      - vmin: the minimum value of the range used for normalization. (Default: value minimum)
      - vmax: the maximum value of the range used for normalization. (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'.(Default: 'gray')
    
    Returns a 3D tensor of shape [batch_size,height, width,3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # quantize
    indices = tf.to_int32(tf.round(value[:,:,:,0]*255))

    # gather
    color_map = cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = color_map(np.arange(256))[:,:3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value

###PER LOSS RIPROIEZIONE###

def bilinear_sampler(imgs, coords):
	"""
	Construct a new image by bilinear sampling from the input image.
	Points falling outside the source image boundary have value 0.
	Args:
		imgs: source image to be sampled from [batch, height_s, width_s, channels]
		coords: coordinates of source pixels to sample from [batch, height_t,width_t, 2]. height_t/width_t correspond to the dimensions of the outputimage (don't need to be the same as height_s/width_s). The two channels correspond to x and y coordinates respectively.
	Returns:
		A new sampled image [batch, height_t, width_t, channels]
	"""

	def _repeat(x, n_repeats):
		rep = tf.transpose(
			tf.expand_dims(tf.ones(shape=tf.stack([
				n_repeats,
			])), 1), [1, 0])
		rep = tf.cast(rep, 'float32')
		x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
		return tf.reshape(x, [-1])

	with tf.name_scope('image_sampling'):
		coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
		inp_size = tf.shape(imgs)
		coord_size = tf.shape(coords)
		out_size = [coord_size[0],coord_size[1],coord_size[2],inp_size[3]]

		coords_x = tf.cast(coords_x, 'float32')
		coords_y = tf.cast(coords_y, 'float32')

		x0 = tf.floor(coords_x)
		x1 = x0 + 1
		y0 = tf.floor(coords_y)
		y1 = y0 + 1

		y_max = tf.cast(inp_size[1] - 1, 'float32')
		x_max = tf.cast(inp_size[2] - 1, 'float32')
		zero = tf.zeros([1], dtype='float32')

		wt_x0 = x1 - coords_x
		wt_x1 = coords_x - x0
		wt_y0 = y1 - coords_y
		wt_y1 = coords_y - y0

		x0_safe = tf.clip_by_value(x0, zero[0], x_max)
		y0_safe = tf.clip_by_value(y0, zero[0], y_max)
		x1_safe = tf.clip_by_value(x1, zero[0], x_max)
		y1_safe = tf.clip_by_value(y1, zero[0], y_max)

		## indices in the flat image to sample from
		dim2 = tf.cast(inp_size[2], 'float32')
		dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
		base = tf.reshape(_repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1,coord_size[1] * coord_size[2]),[out_size[0], out_size[1], out_size[2], 1])

		base_y0 = base + y0_safe * dim2
		base_y1 = base + y1_safe * dim2
		idx00 = x0_safe + base_y0
		idx01 = x0_safe + base_y1
		idx10 = x1_safe + base_y0
		idx11 = x1_safe + base_y1

		## sample from imgs
		imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
		imgs_flat = tf.cast(imgs_flat, 'float32')
		im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
		im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
		im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
		im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

		w00 = wt_x0 * wt_y0
		w01 = wt_x0 * wt_y1
		w10 = wt_x1 * wt_y0
		w11 = wt_x1 * wt_y1

		output = tf.add_n([
			w00 * im00, w01 * im01,
			w10 * im10, w11 * im11
		])

		return output

def warp_image(img, flow):
	"""
	Given an image and a flow generate the warped image, for stereo img is the right image, flow is the disparity alligned with left
	img: image that needs to be warped
	flow: Generic optical flow or disparity
	"""

	def build_coords(immy):   
		max_height = 2048
		max_width = 2048
		pixel_coords = np.ones((1, max_height, max_width, 2))

		# build pixel coordinates and their disparity
		for i in range(0, max_height):
			for j in range(0, max_width):
				pixel_coords[0][i][j][0] = j
				pixel_coords[0][i][j][1] = i

		pixel_coords = tf.constant(pixel_coords, tf.float32)
		real_height = tf.shape(immy)[1]
		real_width = tf.shape(immy)[2]
		real_pixel_coord = pixel_coords[:,0:real_height,0:real_width,:]
		immy = tf.concat([immy, tf.zeros_like(immy)], axis=-1)
		output = real_pixel_coord - immy

		return output

	coords = build_coords(flow)
	warped = bilinear_sampler(img, coords)
	return warped

def _rescale_tf(img,out_shape):
	"""
	Rescale image using bilinear upsampling
	"""
	#print(out_shape)
	def _build_coords(immy,out_shape):
		batch_size = tf.shape(immy)[0]
		in_height = tf.cast(tf.shape(immy)[1],tf.float32)-1
		in_width = tf.cast(tf.shape(immy)[2],tf.float32)-1

		out_height = out_shape[0]
		out_width = out_shape[1]

		delta_x = in_width/tf.cast(out_width-1,tf.float32)
		delta_y = in_height/tf.cast(out_height-1,tf.float32)

		coord_x = tf.concat([tf.range(in_width-1E-4,delta=delta_x,dtype=tf.float32),[in_width]],axis=0)
		coord_x = tf.expand_dims(coord_x,axis=0)
		coord_x_tile = tf.tile(coord_x,[out_height,1])

		coord_y = tf.concat([tf.range(in_height-1E-4,delta=delta_y,dtype=tf.float32),[in_height]],axis=0)
		coord_y = tf.expand_dims(coord_y,axis=1)
		coord_y_tile = tf.tile(coord_y,[1,out_width])

		coord = tf.stack([coord_x_tile,coord_y_tile],axis=-1)
		#coord = tf.Print(coord,[coord[:,:,0]],summarize=1000)
		coord = tf.expand_dims(coord,axis=0)
		coord = tf.tile(coord,[batch_size,1,1,1])

		return coord
	
	coord = _build_coords(img,out_shape)
	warped = bilinear_sampler(img,coord)
	input_shape = img.get_shape().as_list()
	warped.set_shape([input_shape[0],None,None,input_shape[-1]])
	return warped

def rescale_image(img,out_shape):
	if FULLY_DIFFERENTIABLE:
		return _rescale_tf(img,out_shape)
	else:
		return tf.image.resize_images(img,out_shape,method=tf.image.ResizeMethod.BILINEAR)


def resize_to_prediction(x, pred):
    return rescale_image(x,tf.shape(pred)[1:3])
