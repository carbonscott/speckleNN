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
