import numpy as np
import skimage.measure as sm

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
                k, v = info.split(":")
                if not k in kv_dict: kv_dict[k.strip()] = v.strip()

            # Collect data information...
            if kw_data in line:
                info = line[line.rfind(kw_data) + len(kw_data):]
                k = tuple( info.strip().split(",") )
                if not k in data_dict: data_dict[k] = True

    ret_dict = { "kv" : kv_dict, "data" : tuple(data_dict.keys()) }

    return ret_dict


def get_shape_from_conv2d(size_y, size_x, out_channels, 
                                          kernel_size, 
                                          stride,
                                          padding):
    """ Returns the dimension of the output volumne. """
    size_y_out = (size_y - kernel_size + 2 * padding) // stride + 1
    size_x_out = (size_x - kernel_size + 2 * padding) // stride + 1

    return size_y_out, size_x_out, out_channels


def get_shape_from_pool(size_y, size_x, in_channels, 
                                        kernel_size,
                                        stride):
    """ Return the dimension of the output volumen. """
    size_y_out = (size_y - kernel_size) // stride + 1
    size_x_out = (size_x - kernel_size) // stride + 1

    return size_y_out, size_x_out, in_channels


class ConvVolume:
    """ Derive the output size of a conv net. """

    def __init__(self, size_y, size_x, channels, conv_dict):
        self.size_y      = size_y
        self.size_x      = size_x
        self.channels    = channels
        self.conv_dict   = conv_dict
        self.method_dict = { 'conv' : self._get_shape_from_conv2d, 
                             'pool' : self._get_shape_from_pool    }


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

