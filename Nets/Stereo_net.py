import tensorflow as tf
import abc
from collections import OrderedDict


class StereoNet(object):
    __metaclass__ = abc.ABCMeta
    """
    Meta parent class for all the convnets
    """
    #=======================Static Class Fields=============
    _valid_args = [
        ("split_layer", "name of the layer where the network will be splitted"),
        ("sequence", "flag to use network on a video sequence instead of on single images"),
        ("train_portion", "one among 'BEGIN' or 'END' specify which portion of the network will be trained, respectivelly before and after split"),
        ("is_training", "boolean or placeholder to specify if the network is in train or inference mode")
    ]
    _netName="stereoNet"
    #=====================Static Class Methods==============

    @classmethod
    def getPossibleArsg(cls):
        return cls._valid_args

    #==================PRIVATE METHODS======================
    def __init__(self, **kwargs):
        self._layers = OrderedDict()
        self._disparities = []
        self._placeholders = []
        self._placeholderable = []
        self._trainable_variables = OrderedDict() 
        self._layer_to_var = {}
        self._after_split = False
        print('=' * 50)
        print('Starting Creation of {}'.format(self._netName))
        print('=' * 50)

        args = self._validate_args(kwargs)
        print('Args Validated, setting up graph')

        self._preprocess_inputs(args)
        print('Meta op to preprocess data created')

        self._build_network(args)
        print('Network ready')
        print('=' * 50)

    def _get_placeholder_name(self, name):
        """
        convert a layer name to its placehodler version and return it
        """
        return name + '_placeholder'

    def _add_to_layers(self, name, op):
        """
        Add the layer to the network ones and check if name is among the layer where the network should split, if so create a placeholder and return it, otherways return the real layer. Add teh variables to the trainable colelction 
        Args:
            name: name of the layer that need to be addded to the network collection
            op: tensorflow op 
        """
        self._layers[name] = op

        # extract variables
        scope = '/'.join(op.name.split('/')[0:-1])
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) 
        self._layer_to_var[name] = variables

        if not self._after_split:
            # append layer name among those that can be turned into placeholder
            self._placeholderable.append(name)
        if self._after_split != self._train_beginning:  # XOR
            # add variables in scope to the one that needs to be trained
            for v in variables:
                self._trainable_variables[v] = True

        if name in self._split_layers_list:
            # enable flag to mark the first layer after split
            self._after_split = True

    def _get_layer_as_input(self, name):
        # Check if a placeholder for this layer already exist
        if self._get_placeholder_name(name) in self._layers:
            return self._layers[self._get_placeholder_name(name)]
        # check if layer need to be transformed into a placeholder
        elif self._after_split and (not self._sequence) and name in self._placeholderable:
            real_op = self._layers[name]
            placeholder_op = tf.placeholder(
                tf.float32, shape=real_op.get_shape())
            self._layers[self._get_placeholder_name(name)] = placeholder_op
            self._placeholders.append((real_op, placeholder_op))
            return self._layers[self._get_placeholder_name(name)]
        # check if real layer exist
        elif name in self._layers:
            return self._layers[name]
        else:
            raise Exception('Trying to fetch an unknown layer!')

    def __str__(self):
        """to string method"""
        ss = ""
        for k, l in self._layers.items():
            if l in self._disparities:
                ss += "Prediction Layer {}: {}\n".format(k, str(l.shape))
            else:
                ss += "Layer {}: {}\n".format(k, str(l.shape))
        return ss

    def __repr__(self):
        """to string method"""
        return self.__str__()

    def __getitem__(self, key):
        """
        Returns a layer by name
        """
        return self._layers[key]

    #========================ABSTRACT METHODs============================
    @abc.abstractmethod
    def _preprocess_inputs(self, args):
        """
        Abstract method to create metaop that preprocess data before feeding them in the network
        """
        pass

    @abc.abstractmethod
    def _build_network(self, args):
        """
        Should build the elaboration graph
        """
        pass

    @abc.abstractmethod
    def _validate_args(self, args):
        """
        Should validate the argument and add default values
        """
        portion_options = ['BEGIN', 'END']
        # Check common args
        if 'split_layers' not in args:
            print(
                'WARNING: no split points selected, the network will flow without interruption')
            args['split_layers'] = [None]
        if 'train_portion' not in args:
            print('WARNING: train_portion not specified, using default END')
            args['train_portion'] = 'END' if args['split_layers'] != [
                None] else 'BEGIN'
        elif args['train_portion'] not in portion_options:
            raise Exception('Invalid portion options {}'.format(
                args['train_portion']))
        if 'sequence' not in args:
            print('WARNING: sequence flag not setted, configuring the network for single image adaptation')
            args['sequence'] = False
        if 'is_training' not in args:
            print('WARNING: flag for trainign not setted, using default False')
            args['is_training']=False

        # save args value
        self._split_layers_list = args['split_layers']
        self._train_beginning = (args['train_portion'] == 'BEGIN')
        self._sequence = args['sequence']
        self._isTraining=False

    #==============================PUBLIC METHODS==================================
    def get_placeholders(self):
        """
        Get all the placeholder defined internally in the network
        Returns:
            list of couples of layers that became placeholder, each couple is (real_layer,placeholder)
        """
        return self._placeholders

    def get_placeholder(self, name):
        """
        Return the placeholder corresponding to the layer named name
        Args:
            name of the layer where there should be a placeholder
        Returns:
            placeholder for the layer
        """
        placeholder_name = self._get_placeholder_name(name)
        if placeholder_name not in self._layers:
            raise Exception(
                'Unable to find placeholder for layer {}'.format(placeholder_name))
        else:
            return self._layers[placeholder_name]

    def get_all_layers(self):
        """
        Returns all network layers
        """
        return self._layers
    
    def get_layers_names(self):
        """
        Returns all layers name
        """
        return self._layers.keys()

    def get_disparities(self):
        """
        Return all the disparity predicted with increasing resolution
        """
        return self._disparities

    def get_trainable_variables(self):
        """
        Returns the list of trainable variables
        """
        return list(self._trainable_variables.keys())

    def get_variables(self, layer_name):
        """
        Returns the colelction of variables associated to layer_name
        Args:
        layer_name: name of the layer for which we want to access variables
        """
        if layer_name in self._layers and layer_name not in self._layer_to_var:
            return []
        else:
            return self._layer_to_var[layer_name]
