import tensorflow as tf
import numpy as np
import argparse
import Nets
import os
import sys
import time
import cv2
import json
import datetime
import shutil
from matplotlib import pyplot as plt
from Data_utils import data_reader,weights_utils,preprocessing
from Losses import loss_factory
from Sampler import sampler_factory


#static params
PIXEL_TH = 3
MAX_DISP = 192

def main(args):
	#read input data
	with tf.name_scope('input_reader'):
		with tf.name_scope('training_set_reader'):
			data_set = data_reader.dataset(
				args.trainingSet,
				batch_size = args.batchSize,
				crop_shape=args.imageShape,
				num_epochs=args.numEpochs,
				augment=args.augment,
				is_training=True,
				shuffle=True
			)
			left_img_batch, right_img_batch, gt_image_batch = data_set.get_batch()
			inputs={
				'left':left_img_batch,
				'right':right_img_batch,
				'target':gt_image_batch
			}
		if args.validationSet is not None:
			with tf.name_scope('validation_set_reader'):
				validation_set = data_reader.dataset(
					args.validationSet,
					batch_size = args.batchSize,
					augment=False,
					is_training=False,
					shuffle=True
				)
				left_val_batch, right_val_batch, gt_val_batch = validation_set.get_batch()
				print(left_val_batch.shape,right_val_batch.shape)

	#build network
	with tf.variable_scope('model') as scope:
		net_args = {}
		net_args['left_img'] = left_img_batch
		net_args['right_img'] = right_img_batch
		net_args['split_layers'] = [None]
		net_args['sequence'] = True
		net_args['train_portion'] = 'BEGIN'
		net_args['bulkhead'] = False
		stereo_net = Nets.get_stereo_net(args.modelName, net_args)
		print('Stereo Prediction Model:\n', stereo_net)
		predictions = stereo_net.get_disparities()
		full_res_disp = predictions[-1]

		if args.validationSet is not None:
			scope.reuse_variables()
			net_args['left_img']=left_val_batch
			net_args['right_img']=right_val_batch
			val_stereo_net = Nets.get_stereo_net(args.modelName, net_args)
			val_prediction = val_stereo_net.get_disparities()[-1]
	
	if args.validationSet is not None:
		#build validation ops
		with tf.variable_scope('validation_error'):
			# compute error against gt
			abs_err = tf.abs(val_prediction - gt_val_batch)
			valid_map = tf.where(tf.equal(gt_val_batch, 0), tf.zeros_like(gt_val_batch, dtype=tf.float32), tf.ones_like(gt_val_batch, dtype=tf.float32))
			filtered_error = abs_err * valid_map

			abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
			bad_pixel_abs = tf.where(tf.greater(filtered_error, PIXEL_TH), tf.ones_like(filtered_error, dtype=tf.float32), tf.zeros_like(filtered_error, dtype=tf.float32))
			bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)

			tf.summary.scalar('EPE',abs_err)
			tf.summary.scalar('bad3',bad_pixel_perc)
			tf.summary.image('val_prediction',preprocessing.colorize_img(val_prediction,cmap='jet'),max_outputs=1)
			tf.summary.image('val_gt',preprocessing.colorize_img(gt_val_batch,cmap='jet'),max_outputs=1)
	
	with tf.name_scope('training_error'):
		#build train ops
		global_step = tf.Variable(0,trainable=False)
		learning_rate = tf.train.exponential_decay(args.lr,global_step,args.decayStep, 0.5, staircase=True)
		disparity_trainer = tf.train.AdamOptimizer(args.lr,0.9)
		
		#l1 regression loss for each scale mutiplied by the corresponding weight
		if args.lossWeights is not None and len(args.lossWeights)==len(predictions):
			raise ValueError('Wrong number of loss weights provide, should provide {}'.format(len(predictions)))
		full_reconstruction_loss = loss_factory.get_supervised_loss(args.lossType,multiScale=True,logs=False,weights=args.lossWeights,max_disp=MAX_DISP)(predictions,inputs)

		train_op = disparity_trainer.minimize(full_reconstruction_loss,global_step=global_step)

		#add summaries
		tf.summary.image('full_res_disp',preprocessing.colorize_img(full_res_disp,cmap='jet'),max_outputs=1)
		tf.summary.image('gt_disp',preprocessing.colorize_img(gt_image_batch,cmap='jet'),max_outputs=1)
		tf.summary.scalar('full_reconstruction_loss',full_reconstruction_loss)

	#create summary logger
	summary_op = tf.summary.merge_all()
	logger = tf.summary.FileWriter(args.output)

	#create saver
	main_saver = tf.train.Saver(max_to_keep=2)

	#start session
	gpu_options = tf.GPUOptions(allow_growth=True)
	max_steps = data_set.get_max_steps()
	exec_time=0
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		#init stuff
		sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

		#restore disparity inference weights
		restored,step_eval = weights_utils.check_for_weights_or_restore_them(args.output,sess,initial_weights=args.weights)
		print('Disparity Net Restored?: {} from step {}'.format(restored,step_eval))

		sess.run(global_step.assign(step_eval))
		try:	
			start_time = time.time()
			while True:
				tf_fetches = [global_step,train_op,full_reconstruction_loss]

				if step_eval%100==0:
					#summaries
					tf_fetches = tf_fetches + [summary_op]

				#run network
				run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
				fetches = sess.run(tf_fetches,options=run_options)
				
				if step_eval%100==0:
					#log on terminal
					fbTime = (time.time()-start_time)
					exec_time += fbTime
					fbTime = fbTime/100
					logger.add_summary(fetches[-1],global_step=step_eval)
					missing_time=(max_steps-step_eval)*fbTime
					print('Step:{:4d}\tLoss:{:.2f}\tf/b time:{:3f}\tMissing time:{}'.format(step_eval,fetches[2],fbTime,datetime.timedelta(seconds=missing_time)))
					start_time = time.time()
				
				if step_eval%10000==0:
					ckpt = os.path.join(args.output,'weights.ckpt')
					main_saver.save(sess,ckpt,global_step=step_eval)

				step_eval = fetches[0]
		except tf.errors.OutOfRangeError:
			pass
		finally:	
			print('All Done, Bye Bye!')

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Script for training of a Deep Stereo Network')
	parser.add_argument("--trainingSet", help='path to the list file with training set', required=True)
	parser.add_argument("--validationSet", help="path to the list file with the validation set", default=None, type=str)
	parser.add_argument("-o","--output", help="path to the output folder where the results will be saved", required=True)
	parser.add_argument("--weights",help="path to the initial weights for the disparity estimation network (OPTIONAL)")
	parser.add_argument("--modelName", help="name of the stereo model to be used", default="Dispnet", choices=Nets.STEREO_FACTORY.keys())
	parser.add_argument("--lr", help="initial value for learning rate",default=0.0001, type=float)
	parser.add_argument("--imageShape", help='two int for the size of the crop extracted from each image [height,width]', nargs='+', type=int, default=[320,1216])
	parser.add_argument("--batchSize", help='batch size to use during training',type=int,default=4)
	parser.add_argument("--numEpochs", help='number of training epochs',type=int,default=50)
	parser.add_argument("--augment", help="flag to enable data augmentation", action='store_true')
	parser.add_argument("--lossWeights", help="weights for loss at different resolution from full to lower res", nargs='+', default=None, type=float)
	parser.add_argument('--lossType', help="Type of supervised loss to use", choices=loss_factory.SUPERVISED_LOSS.keys(), default="mean_l1",type=str)
	parser.add_argument("--decayStep", help="halve learning rate after this many steps", type=int, default=500000)
	args=parser.parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)

	with open(os.path.join(args.output, 'params.sh'), 'w+') as out:
		sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
		out.write('#!/bin/bash\n')
		out.write('python3 ')
		out.write(' '.join(sys.argv))
		out.write('\n')
	main(args)
