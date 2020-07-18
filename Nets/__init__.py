import Nets.DispNet
import Nets.MadNet
import Nets.PSMNet

STEREO_FACTORY = {
    Nets.DispNet.DispNet._netName: Nets.DispNet.DispNet,
    Nets.MadNet.MadNet._netName: Nets.MadNet.MadNet,
    Nets.PSMNet.PSMNet._netName: Nets.PSMNet.PSMNet
}

def get_stereo_net(name,args):
    if name not in STEREO_FACTORY:
        raise Exception('Unrecognized network name: {}'.format(name))
    else:
        return STEREO_FACTORY[name](**args)