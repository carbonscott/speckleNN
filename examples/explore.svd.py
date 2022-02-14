#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psana
import numpy as np
from deepprojection.datasets.experiments import SPIImgDataset, ConfigDataset
import os

# Config the dataset...
is_resize = None
isflat    = True
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                resize         = is_resize,
                                isflat         = isflat,
                                exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ], )

# Load the dataset...
spiimg = SPIImgDataset(config_dataset)

# Estimate the size of an input matrix for svd analysis...
# Get the dimension of an image
size_y, size_x = spiimg.get_img_and_label(0)[0].shape

# Get the number of images
imglabel_list = spiimg.imglabel_list
num_img  = len(imglabel_list)

# Initialize a numpy array
spimat = np.zeros((num_img, size_y * size_x))

# Load image data into the data matrix...
for i, img_info in enumerate(imglabel_list):
    print(f"Load {img_info}...")
    ## img, _ = spiimg.get_img_and_label(0)
    img, _ = spiimg[i]
    img    = img.reshape(-1)
    spimat[i,:] = img

# SVD...
u, s, vh = np.linalg.svd( spimat.T, full_matrices = False )

# Export to npy...
drc_npy = "npys"
drc_cwd = os.getcwd()
prefixpath_npy = os.path.join(drc_cwd, drc_npy)
if not os.path.exists(prefixpath_npy): os.makedirs(prefixpath_npy)

path_u  = os.path.join(prefixpath_npy, "u.npy")
path_s  = os.path.join(prefixpath_npy, "s.npy")
path_vh = os.path.join(prefixpath_npy, "vh.npy")
np.save(path_u, u)
np.save(path_s, s)
np.save(path_vh, vh)

# Save image dimension for visualization downstream...
path_size_y = os.path.join(prefixpath_npy, "size_y.npy")
path_size_x = os.path.join(prefixpath_npy, "size_x.npy")
np.save(path_size_y, size_y)
np.save(path_size_x, size_x)

# Save the metadata
path_imglabel_list = os.path.join(prefixpath_npy, "imglabel_list.npy")
np.save(path_imglabel_list, imglabel_list)
