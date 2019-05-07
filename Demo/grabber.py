import threading
import time
import numpy as np
import abc

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