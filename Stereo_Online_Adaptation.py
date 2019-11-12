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
MAX_DISP=256
PIXEL_TH = 3

def scale_tensor(tensor,scale):
	return preprocessing.rescale_image(tensor,[tf.shape(tensor)[1]//scale,tf.shape(tensor)[2]//scale])

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x) / np.sum(np.exp(x), axis=0)


def main(args):
	#load json file config
	with open(args.blockConfig) as json_data:
		train_config = json.load(json_data)
	
	#read input data
	with tf.variable_scope('input_reader'):
		data_set = data_reader.dataset(
			args.list,
			batch_size = 1,
			crop_shape=args.imageShape,
			num_epochs=1,
			augment=False,
			is_training=False,
			shuffle=False
		)
		left_img_batch, right_img_batch, gt_image_batch = data_set.get_batch()
		inputs={
			'left':left_img_batch,
			'right':right_img_batch,
			'target':gt_image_batch
		}

	#build inference network
	with tf.variable_scope('model'):
		net_args = {}
		net_args['left_img'] = left_img_batch
		net_args['right_img'] = right_img_batch
		net_args['split_layers'] = [None]
		net_args['sequence'] = True
		net_args['train_portion'] = 'BEGIN'
		net_args['bulkhead'] = True if args.mode=='MAD' else False
		stereo_net = Nets.get_stereo_net(args.modelName, net_args)
		print('Stereo Prediction Model:\n', stereo_net)
		predictions = stereo_net.get_disparities()
		full_res_disp = predictions[-1]
	
	#build real full resolution loss
	with tf.variable_scope('full_res_loss'):
		# reconstruction loss between warped right image and original left image
		full_reconstruction_loss =  loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)(predictions,inputs)


	#build validation ops
	with tf.variable_scope('validation_error'):
		# compute error against gt
		abs_err = tf.abs(full_res_disp - gt_image_batch)
		valid_map = tf.where(tf.equal(gt_image_batch, 0), tf.zeros_like(gt_image_batch, dtype=tf.float32), tf.ones_like(gt_image_batch, dtype=tf.float32))
		filtered_error = abs_err * valid_map

		abs_err = tf.reduce_sum(filtered_error) / tf.reduce_sum(valid_map)
		bad_pixel_abs = tf.where(tf.greater(filtered_error, PIXEL_TH), tf.ones_like(filtered_error, dtype=tf.float32), tf.zeros_like(filtered_error, dtype=tf.float32))
		bad_pixel_perc = tf.reduce_sum(bad_pixel_abs) / tf.reduce_sum(valid_map)
	
	#build train ops
	disparity_trainer = tf.train.MomentumOptimizer(args.lr,0.9)
	train_ops = []
	if args.mode == 'MAD':
		#build train ops for separate portion of the network
		predictions = predictions[:-1] #remove full res disp
		
		inputs_modules = {
			'left':scale_tensor(left_img_batch,args.reprojectionScale),
			'right':scale_tensor(right_img_batch,args.reprojectionScale),
			'target':scale_tensor(gt_image_batch,args.reprojectionScale)/args.reprojectionScale
		}
		
		assert(len(predictions)==len(train_config))
		for counter,p in enumerate(predictions):
			print('Build train ops for disparity {}'.format(counter))
					
			#rescale predictions to proper resolution
			multiplier = tf.cast(tf.shape(left_img_batch)[1]//tf.shape(p)[1],tf.float32)
			p = preprocessing.resize_to_prediction(p,inputs_modules['left'])*multiplier

			#compute reprojection error
			with tf.variable_scope('reprojection_'+str(counter)):
				reconstruction_loss = loss_factory.get_reprojection_loss('mean_SSIM_l1',reduced=True)([p],inputs_modules)

			#build train op
			layer_to_train = train_config[counter]
			print('Going to train on {}'.format(layer_to_train))
			var_accumulator=[]
			for name in layer_to_train:
				var_accumulator+=stereo_net.get_variables(name)
			print('Number of variable to train: {}'.format(len(var_accumulator)))
				
			#add new training op
			train_ops.append(disparity_trainer.minimize(reconstruction_loss,var_list=var_accumulator))

			print('Done')
			print('='*50)
		
		#create Sampler to fetch portions to train
		sampler = sampler_factory.get_sampler(args.sampleMode,args.numBlocks,args.fixedID)
		
	elif args.mode=='FULL':
		#build single train op for the full network
		train_ops.append(disparity_trainer.minimize(full_reconstruction_loss))
	

	if args.summary:
		#add summaries
		tf.summary.scalar('EPE',abs_err)
		tf.summary.scalar('bad3',bad_pixel_perc)
		tf.summary.image('full_res_disp',preprocessing.colorize_img(full_res_disp,cmap='jet'),max_outputs=1)
		tf.summary.image('gt_disp',preprocessing.colorize_img(gt_image_batch,cmap='jet'),max_outputs=1)

		#create summary logger
		summary_op = tf.summary.merge_all()
		logger = tf.summary.FileWriter(args.output)


	#start session
	gpu_options = tf.GPUOptions(allow_growth=True)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		#init stuff
		sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

		#restore disparity inference weights
		var_to_restore = weights_utils.get_var_to_restore_list(args.weights, [])
		assert(len(var_to_restore)>0)
		restorer = tf.train.Saver(var_list=var_to_restore)
		restorer.restore(sess,args.weights)
		print('Disparity Net Restored?: {}, number of restored variables: {}'.format(True,len(var_to_restore)))

		num_actions=len(train_ops)
		if args.mode=='FULL':
			selected_train_ops = train_ops
		else:
			selected_train_ops = [tf.no_op()]

		epe_accumulator = []
		bad3_accumulator = []
		time_accumulator = []
		exec_time = 0
		fetch_counter=[0]*num_actions
		sample_distribution=np.zeros(shape=[num_actions])
		temp_score = np.zeros(shape=[num_actions])
		loss_t_2 = 0
		loss_t_1 = 0
		expected_loss = 0
		last_trained_blocks = []
		reset_counter=0
		step=0
		max_steps=data_set.get_max_steps()
		try:	
			start_time = time.time()
			while True:

				#fetch new network portion to train
				if step%args.sampleFrequency==0 and args.mode=='MAD':
					#Sample 
					distribution = softmax(sample_distribution)
					blocks_to_train = sampler.sample(distribution)
					selected_train_ops = [train_ops[i] for i in blocks_to_train]

					#accumulate sampling statistics
					for l in blocks_to_train:
						fetch_counter[l]+=1

				#build list of tensorflow operations that needs to be executed

				#errors and full resolution loss
				tf_fetches = [abs_err,bad_pixel_perc,full_reconstruction_loss]

				if args.summary and step%100==0:
					#summaries
					tf_fetches = tf_fetches + [summary_op]

				#update ops
				tf_fetches = tf_fetches+selected_train_ops

				if args.logDispStep!=-1 and step%args.logDispStep==0:
					#prediction for serialization to disk
					tf_fetches=tf_fetches + [full_res_disp]

				#run network
				fetches = sess.run(tf_fetches)
				new_loss = fetches[2]

				if args.mode == 'MAD':
					#update sampling probabilities
					if step==0:
						loss_t_2 = new_loss
						loss_t_1 = new_loss
					expected_loss = 2*loss_t_1-loss_t_2	
					gain_loss=expected_loss-new_loss
					sample_distribution = 0.99*sample_distribution
					for i in last_trained_blocks:
						sample_distribution[i] += 0.01*gain_loss

					last_trained_blocks=blocks_to_train
					loss_t_2 = loss_t_1
					loss_t_1 = new_loss
					
				#accumulate performance metrics
				epe_accumulator.append(fetches[0])
				bad3_accumulator.append(fetches[1])

				if step%100==0:
					#log on terminal
					fbTime = (time.time()-start_time)
					exec_time += fbTime
					fbTime = fbTime/100
					if args.summary:
						logger.add_summary(fetches[3],global_step=step)
					missing_time=(max_steps-step)*fbTime
					print('Step:{:4d}\tbad3:{:.2f}\tEPE:{:.2f}\tSSIM:{:.2f}\tf/b time:{:3f}\tMissing time:{}'.format(step,fetches[1], fetches[0],new_loss,fbTime,datetime.timedelta(seconds=missing_time)))
					start_time = time.time()
				
				#reset network if necessary
				if new_loss>args.SSIMTh:
					restorer.restore(sess,args.weights)
					reset_counter+=1
				
				#save disparity if requested
				if args.logDispStep!=-1 and step%args.logDispStep==0:
					dispy=fetches[-1]
					dispy_to_save = np.clip(dispy[0], 0, MAX_DISP)
					dispy_to_save = (dispy_to_save*256.0).astype(np.uint16)
					cv2.imwrite(os.path.join(args.output, 'disparities/disparity_{}.png'.format(step)), dispy_to_save)

				step+=1

		except tf.errors.OutOfRangeError:
			pass
		finally:
			epe_array = epe_accumulator
			bad3_array = bad3_accumulator
			epe_accumulator = np.sum(epe_accumulator)
			bad3_accumulator = np.sum(bad3_accumulator)
			with open(os.path.join(args.output, 'stats.csv'), 'w+') as f_out:
				# report series
				f_out.write('Metrics,cumulative,average\n')
				f_out.write('EPE,{},{}\n'.format(epe_accumulator,epe_accumulator/step))
				f_out.write('bad3,{},{}\n'.format(bad3_accumulator,bad3_accumulator/step))
				f_out.write('time,{},{}\n'.format(exec_time,exec_time/step))
				f_out.write('FPS,{}\n'.format(1/(exec_time/step)))
				f_out.write('#resets,{}\n'.format(reset_counter))
				f_out.write('Blocks')
				for n in range(len(predictions)):
					f_out.write(',{}'.format(n))
				f_out.write(',final\n')
				f_out.write('fetch_counter')
				for c in fetch_counter:
					f_out.write(',{}'.format(c))
				f_out.write('\n')
				for c in sample_distribution:
					f_out.write(',{}'.format(c))
				f_out.write('\n')
			
			step_time = exec_time/step
			time_array = [str(x*step_time) for x in range(len(epe_array))]
			
			with open(os.path.join(args.output,'series.csv'),'w+') as f_out:
				f_out.write('Iteration,Time,EPE,bad3\n')
				for i,(t,e,b) in enumerate(zip(time_array,epe_array,bad3_array)):
					f_out.write('{},{},{},{}\n'.format(i,t,e,b))
			
			print('Result saved in {}'.format(args.output))
			
			print('All Done, Bye Bye!')

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Script for online Adaptation of a Deep Stereo Network')
	parser.add_argument("-l","--list", help='path to the list file with frames to be processed', required=True)
	parser.add_argument("-o","--output", help="path to the output folder where the results will be saved", required=True)
	parser.add_argument("--weights",help="path to the initial weights for the disparity estimation network",required=True)
	parser.add_argument("--modelName", help="name of the stereo model to be used", default="Dispnet", choices=Nets.STEREO_FACTORY.keys())
	parser.add_argument("--numBlocks", help="number of CNN portions to train at each iteration",type=int,default=1)
	parser.add_argument("--lr", help="value for learning rate",default=0.0001, type=float)
	parser.add_argument("--blockConfig",help="path to the block_config json file",required=True)
	parser.add_argument("--sampleMode",help="choose the sampling heuristic to use",choices=sampler_factory.AVAILABLE_SAMPLER,default='SAMPLE')
	parser.add_argument("--fixedID",help="index of the portions of network to train, used only if sampleMode=FIXED",type=int,nargs='+',default=[0])
	parser.add_argument("--reprojectionScale",help="compute all loss function at 1/reprojectionScale",default=1,type=int)
	parser.add_argument("--summary",help='flag to enable tensorboard summaries',action='store_true')
	parser.add_argument("--imageShape", help='two int for the size of the crop extracted from each image [height,width]', nargs='+', type=int, default=[320,1216])
	parser.add_argument("--SSIMTh",help="reset network to initial configuration if loss is above this value",type=float,default=0.5)
	parser.add_argument("--sampleFrequency",help="sample new network portions to train every K frame",type=int,default=1)
	parser.add_argument("--mode",help="online adaptation mode: NONE - perform only inference, FULL - full online backprop, MAD - backprop only on portions of the network", choices=['NONE','FULL','MAD'], default='MAD')
	parser.add_argument("--logDispStep", help="save disparity every K step, -1 to disable", default=-1, type=int)
	args=parser.parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)
	if args.logDispStep!=-1 and not os.path.exists(os.path.join(args.output, 'disparities')):
		os.makedirs(os.path.join(args.output, 'disparities'))
	shutil.copy(args.blockConfig,os.path.join(args.output,'config.json'))
	with open(os.path.join(args.output, 'params.sh'), 'w+') as out:
		sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
		out.write('#!/bin/bash\n')
		out.write('python3 ')
		out.write(' '.join(sys.argv))
		out.write('\n')
	main(args)
