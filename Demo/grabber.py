import threading
import time
import numpy as np
import abc
import json

############################################################
###########          CAMERA FACTORY              ###########
###########################################################

_GRABBER_FACTORY={}

def get_camera(name, frame_queue, config=None, framerate=30):
    """
    factory method to construct a camera wrapper
    """
    if name not in _GRABBER_FACTORY:
        raise Exception('Unrecognized camera type: {}'.format(name))
    else:
        return _GRABBER_FACTORY[name](frame_queue, config=config, framerate=framerate)

def get_available_camera():
    return _GRABBER_FACTORY.keys()

def register_camera_to_factory():
    def decorator(cls):
        _GRABBER_FACTORY[cls._name]=cls
        return cls
    return decorator

#####################################################################
##############           ABSTRACT CAMERA                #############
####################################################################


class ImageGrabber(threading.Thread):
    """
    Thread to grab frames from the camera and load them in frame_queue
    """
    __metaclass__=abc.ABCMeta
    def __init__(self, frame_queue, config=None, framerate=30):
        """
        Args
        -----
        frame_queue: queue
            synchronized queue where left and right frames will be loaded
        config: path
            path to json file for calibration and/or other configuration parameters
        framerate: int
            target frame per second
        """
        threading.Thread.__init__(self)
        self._config = config
        self._connect_to_camera()
        self._buffer = frame_queue
        self._sleeptime = 1/framerate
        self._stop_acquire=False
    
    def stop(self):
        """
        Stop the acquisition of new frames from the camera and kill the thread
        """
        self._stop_acquire=True
    
    def run(self):
        """
        Main body method, grab frames from camera and put them on buffer as a [2,h,w,c] numpy array
        """
        while not self._stop_acquire:
            l,r = self._read_frame()
            self._buffer.put(np.stack([l,r],axis=0))
            time.sleep(self._sleeptime)
        
        self._disconnect_from_camera()

    @abc.abstractmethod
    def _read_frame(self):
        """
        Read left and right rectified frame and return them
        """
    
    @abc.abstractmethod
    def _connect_to_camera(self):
        """
        Connect to external camera
        """
    
    @abc.abstractmethod
    def _disconnect_from_camera(self):
        """
        Disconnect from external camera
        """


#########################################################################
#################           ZED MINI                    #################
#########################################################################

import pyzed.sl as sl

@register_camera_to_factory()
class ZEDMini(ImageGrabber):
    _name = 'ZED_Mini'
    _key_to_res = {
        '2K' : sl.RESOLUTION.RESOLUTION_HD2K,
        '1080p' : sl.RESOLUTION.RESOLUTION_HD1080,
        '720p' : sl.RESOLUTION.RESOLUTION_HD720,
        'VGA' : sl.RESOLUTION.RESOLUTION_VGA
    }

    """ Read Stereo frames from a ZED Mini stereo camera. """
    def _read_frame(self):
        err = self._cam.grab(self._runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self._cam.retrieve_image(self._left_frame, sl.VIEW.VIEW_LEFT)
            self._cam.retrieve_image(self._right_frame, sl.VIEW.VIEW_RIGHT)
            return self._left_frame.get_data()[:,:,:3], self._right_frame.get_data()[:,:,:3]
    
    def _connect_to_camera(self):
        # road option from config file
        with open(self._config) as f_in:
            self._config = json.load(f_in)

        self._params = sl.InitParameters()
        
        if 'resolution' in self._config:
            self._params.camera_resolution = self._key_to_res[self._config['resolution']]
        else:
            self._params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
        
        if 'fps' in self._config:
            self._params.camera_fps = self._config['fps']
        else:
            self._params.camera_fps = 30
        
        self._cam = sl.Camera()
        status = self._cam.open(self._params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(status)
            raise Exception('Unable to connect to Stereo Camera')
        self._runtime = sl.RuntimeParameters()
        self._left_frame = sl.Mat()
        self._right_frame = sl.Mat()

    def _disconnect_from_camera(self):
        self._cam.close()        


#########################################################################
#################           SMATT CAM                   #################
#########################################################################

#Example of frame grabber for a custom camera

# from stereocam import StereoCamera 

# @register_camera_to_factory()
# class SmattCam(ImageGrabber):
#     _name='SmattCam'
#     """
#     Read frames from smart camera from Mattoccia et al.
#     """
#     def _read_frame(self):
#         left,right =  self._cam.grab_frames()
#         left = np.repeat(left, 3, axis=-1)
#         right = np.repeat(right, 3, axis=-1)
#         return left,right
    
#     def _connect_to_camera(self):
#         self._cam = StereoCamera(self._config)
#         self._cam.calibrate()

    
#     def _disconnect_from_camera(self):
#         pass