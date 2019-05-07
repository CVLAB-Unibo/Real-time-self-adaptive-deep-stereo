import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import Queue
import argparse
import grabber
import cv2
import demo_model
import time

import Nets


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Real-time stereo live demo')
    parser.add_argument("--modelName", help="name of the stereo model to be used", default="MADNet", choices=Nets.STEREO_FACTORY.keys())
    parser.add_argument("--weights",help="path to the initial weights for the disparity estimation network",default=None)
    parser.add_argument("--mode",help="online adaptation mode: NONE - perform only inference, FULL - full online backprop, MAD - backprop only on portions of the network", choices=['NONE','FULL','MAD'], default='NONE')
    parser.add_argument("--lr", help="value for learning rate",default=0.0001, type=float)
    parser.add_argument("--blockConfig",help="path to the block_config json file",default='../block_config/MadNet_full.json')
    parser.add_argument("--imageShape", help='two int for image shape [height,width]', nargs='+', type=int, default=[480,640])
    parser.add_argument("--cropShape", help='two int for crop shape [height,width]', nargs='+', type=int, default=[320,512]), 
    parser.add_argument("--SSIMTh",help="reset network to initial configuration if loss is above this value",type=float,default=0.5)
    parser.add_argument("--cameraConfig", help="path to a configuration file for the camera", default='/home/alessio/code/Real-time-self-adaptive-deep-stereo/Demo/configuration.json')
    args = parser.parse_args()

    assert(os.path.exists(args.cameraConfig))
    assert(len(args.imageShape)==2)

    #setup shared queue for readed frame
    camera_frames = Queue.Queue(1)

    #create camera grabber and disparity network
    dd = demo_model.RealTimeStereo(
        camera_frames,
        model_name=args.modelName,
        weight_path=args.weights,
        learning_rate=args.lr,
        block_config_path=args.blockConfig,
        image_shape = args.imageShape,
        crop_shape=args.cropShape,
        SSIMTh = args.SSIMTh,
        mode = args.mode
        )
    gg = grabber.get_camera(
        'SmattCam',
        camera_frames,
        config=args.cameraConfig, 
        framerate=30)

    print('Threads ready to start')
    #unleash the thread
    gg.start()
    dd.start()

    #print('Going To sleep')
    a=raw_input('Press something to stop')
    
    print('Requesting Stops')

    gg.stop()
    gg.join()
    print('Camera grabber stopped')


    dd.stop()
    dd.join()
    print('detector stopped')

    print('Goodbye')