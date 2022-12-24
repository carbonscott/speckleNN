#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import tqdm
import skimage.measure as sm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return None




class EpochManager:

    def __init__(self, trainer, validator):

        self.trainer   = trainer
        self.validator = validator

        # Track model training information
        self.param_update_ratio_dict  = {}
        self.activation_dict          = {}
        self.preactivation_dict       = {}
        self.model_named_parameters   = []
        self.model_named_gradients    = []

        # Track state variable
        self.loss_min = float('inf')

        return None


    def save_state_dict(self):
        timestamp = self.trainer.config.timestamp

        DRCDEBUG = "debug"
        drc_cwd = os.getcwd()
        prefixpath_debug = os.path.join(drc_cwd, DRCDEBUG)
        if not os.path.exists(prefixpath_debug): os.makedirs(prefixpath_debug)
        fl_debug   = f"{timestamp}.train.debug"
        path_debug = os.path.join(prefixpath_debug, fl_debug)

        # Hmmm, DataParallel wrappers keep raw model object in .module attribute
        state_dict = {
            "param_update"           : self.param_update_ratio_dict,
            "preactivation"          : self.preactivation_dict,
            "activation"             : self.activation_dict,
            "model_named_parameters" : self.model_named_parameters,
            "model_named_gradients"  : self.model_named_gradients,
        }
        torch.save(state_dict, path_debug)


    def save_model_parameters(self):
        self.model_named_parameters = list(self.trainer.model.named_parameters())


    def save_model_gradients(self):
        self.model_named_gradients = [ (name, param.grad) for name, param in self.trainer.model.named_parameters() ]


    def run_one_epoch(self, epoch, returns_loss = False, logs_batch_loss = False):
        # Run one epoch of training...
        loss_train = self.trainer.train(epoch = epoch, returns_loss = True, logs_batch_loss = logs_batch_loss)

        # Pass the model to validator for immediate validation...
        self.validator.model = self.trainer.model

        # Run one epoch of training...
        loss_validate = self.validator.validate(returns_loss = True, epoch = epoch, logs_batch_loss = logs_batch_loss)

        # Save checkpoint whenever validation loss gets smaller...
        # Notice it doesn't imply early stopping
        if loss_validate < self.loss_min:
            # Save a checkpoint file...
            self.trainer.save_checkpoint(epoch = epoch)

            # Update the new loss...
            self.loss_min = loss_validate

        return loss_train, loss_validate, self.loss_min if returns_loss else None


    def run(self, max_epochs):
        # Track the min of loss from inf...
        loss_min = float('inf')

        # Start trainig and validation...
        for epoch in tqdm.tqdm(range(max_epochs)):
            # Run one epoch of training...
            self.trainer.train(epoch = epoch)

            # Pass the model to validator for immediate validation...
            self.validator.model = self.trainer.model

            # Run one epoch of training...
            loss_validate = self.validator.validate(returns_loss = True, epoch = epoch)

            # Save checkpoint whenever validation loss gets smaller...
            # Notice it doesn't imply early stopping
            if loss_validate < loss_min: 
                # Save a checkpoint file...
                self.trainer.save_checkpoint(self.timestamp)

                # Update the new loss...
                loss_min = loss_validate

        return None


    def save_param_update_ratio(self):
        with torch.no_grad():
            for name, param in self.trainer.model.named_parameters():
                if param.grad is None: continue

                if name not in self.param_update_ratio_dict: self.param_update_ratio_dict[name] = []

                # Calculate update_ratio at each layer...
                update_ratio = (param.grad.std() / param.std()).item()
                self.param_update_ratio_dict[name] = update_ratio


    def build_layer_hook(self, module_name, tag = ''):
        if tag not in self.preactivation_dict: self.preactivation_dict[tag] = {}
        if tag not in self.activation_dict   : self.activation_dict   [tag] = {}
        def hook(model, input, output):
            self.preactivation_dict[tag][module_name] = input
            self.activation_dict   [tag][module_name] = output
        return hook


    def set_layer_to_capture(self, module_name_capture_list = [], module_layer_capture_list = []):
        for module_name, module_layer in self.trainer.model.named_modules():
            # Capture based on module name...
            for module_name_capture in module_name_capture_list:
                if module_name_capture in module_name:
                    module_layer.register_forward_hook(self.build_layer_hook(module_name, module_name_capture))

            # Capture based on module layer...
            for module_layer_capture in module_layer_capture_list:
                if isinstance(module_layer, module_layer_capture):
                    module_layer.register_forward_hook(self.build_layer_hook(module_name, module_layer_capture))




def init_logger(log_name = 'train', timestamp = None, returns_timestamp = False, saves_log = True):
    if timestamp is None:
        # Create a timestamp to name the log file...
        now = datetime.now()
        timestamp = now.strftime("%Y_%m%d_%H%M_%S")

    # Configure the location to run the job...
    drc_cwd = os.getcwd()

    # Set up the log file...
    fl_log         = f"{timestamp}.{log_name}.log"
    DRCLOG         = "logs"
    prefixpath_log = os.path.join(drc_cwd, DRCLOG)
    if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
    path_log = os.path.join(prefixpath_log, fl_log) if saves_log else None

    # Config logging behaviors
    logging.basicConfig( filename = path_log,
                         filemode = 'w',
                         format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                         datefmt="%m/%d/%Y %H:%M:%S",
                         level=logging.INFO, )
    logger = logging.getLogger(__name__)

    return timestamp if returns_timestamp else None




class ConvVolume:
    """ Derive the output size of a conv net. """

    def __init__(self, size_y, size_x, channels, conv_dict):
        self.size_y      = size_y
        self.size_x      = size_x
        self.channels    = channels
        self.conv_dict   = conv_dict
        self.method_dict = { 'conv' : self._get_shape_from_conv2d, 
                             'pool' : self._get_shape_from_pool    }

        return None


    def shape(self):
        for layer_name in self.conv_dict["order"]:
            # Obtain the method name...
            method, _ = layer_name.split()

            # Unpack layer params...
            layer_params = self.conv_dict[layer_name]

            #  Obtain the size of the new volume...
            self.channels, self.size_y, self.size_x = \
                self.method_dict[method](**layer_params)

        return self.channels, self.size_y, self.size_x


    def _get_shape_from_conv2d(self, **kwargs):
        """ Returns the dimension of the output volumne. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = kwargs["out_channels"]
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]
        padding      = kwargs["padding"]

        out_size_y = (size_y - kernel_size + 2 * padding) // stride + 1
        out_size_x = (size_x - kernel_size + 2 * padding) // stride + 1

        return out_channels, out_size_y, out_size_x


    def _get_shape_from_pool(self, **kwargs):
        """ Return the dimension of the output volumen. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = self.channels
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]

        out_size_y = (size_y - kernel_size) // stride + 1
        out_size_x = (size_x - kernel_size) // stride + 1

        return out_channels, out_size_y, out_size_x




class MetaLog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ MetaLog \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




def downsample(assem, bin_row=2, bin_col=2, mask=None):
    """ Downsample an SPI image.  
        Adopted from https://github.com/chuckie82/DeepProjection/blob/master/DeepProjection/utils.py
    """
    if mask is None:
        combinedMask = np.ones_like(assem)
    else:
        combinedMask = mask
    downCalib  = sm.block_reduce(assem       , block_size=(bin_row, bin_col), func=np.sum)
    downWeight = sm.block_reduce(combinedMask, block_size=(bin_row, bin_col), func=np.sum)
    warr       = np.zeros_like(downCalib, dtype='float32')
    ind        = np.where(downWeight > 0)
    warr[ind]  = downCalib[ind] / downWeight[ind]

    return warr




def read_log(file):
    '''Return all lines in the user supplied parameter file without comments.
    '''
    # Retrieve key-value information...
    kw_kv     = "KV - "
    kv_dict   = {}

    # Retrieve data information...
    kw_data   = "DATA - "
    data_dict = {}
    with open(file,'r') as fh:
        for line in fh.readlines():
            # Collect kv information...
            if kw_kv in line:
                info = line[line.rfind(kw_kv) + len(kw_kv):]
                k, v = info.split(":", maxsplit = 1)
                if not k in kv_dict: kv_dict[k.strip()] = v.strip()

            # Collect data information...
            if kw_data in line:
                info = line[line.rfind(kw_data) + len(kw_data):]
                k = tuple( info.strip().split(",") )
                if not k in data_dict: data_dict[k] = True

    ret_dict = { "kv" : kv_dict, "data" : tuple(data_dict.keys()) }

    return ret_dict


def calc_dmat(emb1_list, emb2_list, is_sqrt = True):
    ''' Return a 2D distance matrix.

        emb1.shape: len(emb1_list), len(emb1_list[0])
        emb2.shape: len(emb2_list), len(emb2_list[0])
    '''
    # Calculate the difference vector...
    # emb1[:, None] has a dim of [ num1, 1   , dim ], equivalent to [num1, num2, dim] by stretching/replicating axis=1 num2 times.  
    # emb2[None, :] has a dim of [ 1   , num2, dim ], equivalent to [num1, num2, dim] by stretching/replicating axis=0 num2 times.  
    # subtraction returns dim of [ num1, num2, dim ]
    delta_distance_vector = emb1_list[:, None] - emb2_list[None, :]

    # Calculate the squared distance matrix...
    dmat = torch.sum( delta_distance_vector * delta_distance_vector, dim = -1 )

    # Apply square-root is necessary???
    if is_sqrt: dmat = torch.sqrt(dmat)

    return dmat




def split_dataset(dataset_list, fracA, seed = None):
    ''' Split a dataset into two subsets A and B by user-specified fraction.
    '''
    # Set seed for data spliting...
    if seed is not None:
        random.seed(seed)

    # Indexing elements in the dataset...
    size_dataset = len(dataset_list)
    idx_dataset = range(size_dataset)

    # Get the size of the dataset and the subset A...
    size_fracA   = int(fracA * size_dataset)

    # Randomly choosing examples for constructing subset A...
    idx_fracA_list = random.sample(idx_dataset, size_fracA)

    # Obtain the subset B...
    idx_fracB_list = set(idx_dataset) - set(idx_fracA_list)
    idx_fracB_list = sorted(list(idx_fracB_list))

    fracA_list = [ dataset_list[idx] for idx in idx_fracA_list ]
    fracB_list = [ dataset_list[idx] for idx in idx_fracB_list ]

    return fracA_list, fracB_list




def split_list_into_chunk(input_list, max_num_chunk = 2):

    chunk_size = len(input_list) // max_num_chunk + 1

    size_list = len(input_list)

    chunked_list = []
    for idx_chunk in range(max_num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        ## if idx_chunk == max_num_chunk - 1: idx_e = len(input_list)
        if idx_e >= size_list: idx_e = size_list

        seg = input_list[idx_b : idx_e]
        chunked_list.append(seg)

        if idx_e == size_list: break

    return chunked_list




def split_dict_into_chunk(input_dict, max_num_chunk = 2):

    chunk_size = len(input_dict) // max_num_chunk + 1

    size_dict = len(input_dict)
    kv_iter   = iter(input_dict.items())

    chunked_dict_in_list = []
    for idx_chunk in range(max_num_chunk):
        chunked_dict = {}
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_e >= size_dict: idx_e = size_dict

        for _ in range(idx_e - idx_b):
            k, v = next(kv_iter)
            chunked_dict[k] = v
        chunked_dict_in_list.append(chunked_dict)

        if idx_e == size_dict: break

    return chunked_dict_in_list




class ConfusionMatrix:
    def __init__(self, res_dict):
        self.res_dict = res_dict


    def reduce_confusion(self, label):
        ''' Given a label, reduce multiclass confusion matrix to binary
            confusion matrix.
        '''
        res_dict    = self.res_dict
        labels      = res_dict.keys()
        labels_rest = [ i for i in labels if not i == label ]

        # Early return if non-exist label is passed in...
        if not label in labels: 
            print(f"label {label} doesn't exist!!!")
            return None

        # Obtain true positive...
        tp = len(res_dict[label][label])
        fp = sum( [ len(res_dict[label][i]) for i in labels_rest ] )
        tn = sum( sum( len(res_dict[i][j]) for j in labels_rest ) for i in labels_rest )
        fn = sum( [ len(res_dict[i][label]) for i in labels_rest ] )

        return tp, fp, tn, fn


    def get_metrics(self, label):
        # Early return if non-exist label is passed in...
        confusion = self.reduce_confusion(label)
        if confusion is None: return None

        # Calculate metrics...
        tp, fp, tn, fn = confusion
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        precision   = tp / (tp + fp)
        recall      = tp / (tp + fn)
        specificity = tn / (tn + fp) if tn + fp > 0 else None
        f1_inv      = (1 / precision + 1 / recall)
        f1          = 2 / f1_inv

        return accuracy, precision, recall, specificity, f1




class TorchModelAttributeParser:

    def __init__(self): return None

    def reset_attr_dict(self):

        module_to_attr_dict = {
            nn.Conv2d : {
                "in_channels"  : None,
                "out_channels" : None,
                "kernel_size"  : None,
                "stride"       : None,
                "padding"      : None,
                "dilation"     : None,
                "groups"       : None,
                "bias"         : None,
                "padding_mode" : None,
                "device"       : None,
                "dtype"        : None,
            },

            nn.BatchNorm2d : {
                "num_features"        : None,
                "eps"                 : None,
                "momentum"            : None,
                "affine"              : None,
                "track_running_stats" : None,
                "device"              : None,
                "dtype"               : None,
            },

            nn.MaxPool2d : {
                "kernel_size"    : None,
                "stride"         : None,
                "padding"        : None,
                "dilation"       : None,
                "return_indices" : None,
                "ceil_mode"      : None,
            },
        }

        return module_to_attr_dict


    def parse(self, model):
        module_to_attr_dict = self.reset_attr_dict()

        model_type = type(model)
        attr_dict  = module_to_attr_dict.get(model_type, {})

        for attr in attr_dict.keys():
            attr_dict[attr] = getattr(model, attr, None)

        return model_type, attr_dict




class NNSize:
    """ Derive the output size of a conv net. """

    def __init__(self, size_y, size_x, channels, conv_dict):
        self.size_y      = size_y
        self.size_x      = size_x
        self.channels    = channels
        self.conv_dict   = conv_dict
        self.method_dict = { nn.Conv2d    : self.get_shape_from_conv2d,
                             nn.MaxPool2d : self.get_shape_from_pool }

        logger.info("___/ NEURAL NETWORK SHAPE \___")

        return None


    def shape(self):
        for layer_name, (model_type, model_attr_tuple) in self.conv_dict.items():
            if model_type not in self.method_dict: continue

            logger.info(f"layer: {layer_name}, {model_type}")
            logger.info(f"in : {self.channels}, {self.size_y}, {self.size_x}")

            #  Obtain the size of the new volume...
            self.channels, self.size_y, self.size_x = \
                self.method_dict[model_type](**model_attr_tuple)

            logger.info(f"out: {self.channels}, {self.size_y}, {self.size_x}")
            logger.info(f"")

        return self.channels, self.size_y, self.size_x


    def get_shape_from_conv2d(self, **kwargs):
        """ Returns the dimension of the output volumne. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = kwargs["out_channels"]
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]
        padding      = kwargs["padding"]

        kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride      = stride[0]      if isinstance(stride     , tuple) else stride
        padding     = padding[0]     if isinstance(padding    , tuple) else padding

        out_size_y = (size_y - kernel_size + 2 * padding) // stride + 1
        out_size_x = (size_x - kernel_size + 2 * padding) // stride + 1

        return out_channels, out_size_y, out_size_x


    def get_shape_from_pool(self, **kwargs):
        """ Return the dimension of the output volumen. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = self.channels
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]

        kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride      = stride[0]      if isinstance(stride     , tuple) else stride

        out_size_y = (size_y - kernel_size) // stride + 1
        out_size_x = (size_x - kernel_size) // stride + 1

        return out_channels, out_size_y, out_size_x




class Config:
    def __init__(self, name, **kwargs):
        logger.info(f"___/ Configure {name} \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16} : {v}")
