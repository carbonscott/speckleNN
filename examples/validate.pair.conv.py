#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch
from deepprojection.datasets.experiments import SPIImgDataset      , SiameseTestset, ConfigDataset
from deepprojection.model                import SiameseModelCompare, ConfigSiameseModel
from deepprojection.validator            import PairValidator      , ConfigValidator
from deepprojection.encoders.convnet     import Hirotaka0122       , ConfigEncoder


# Create a timestamp to name the log file...
## timestamp = "20220207115721"
timestamp = "20220214220141"

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.pair.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Config logging behaviors
logging.basicConfig( filename = path_log,
                     filemode = 'w',
                     format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 2000, 
                                mode           = 'image',
                                mask           = None,
                                resize         = None,
                                isflat         = False,
                                exclude_labels = exclude_labels, )

# Get image size...
spiimg = SPIImgDataset(config_dataset)
img    = spiimg.get_img_and_label(0)[0]
size_y, size_x = img.shape

# Creat a mask...
# Create a raw mask
mask = np.ones_like(img)

# Mask out the top 10%
top = 0.1
h_false = int(top * size_y)
mask_false_area = (slice(0, h_false), slice(0, size_x))
mask[mask_false_area[0], mask_false_area[1]] = 0

# Mask out the oversaturated panel in 102
mask_false_area = (slice(510, None), slice(541, 670))
mask[mask_false_area[0], mask_false_area[1]] = 0

# Reconfig the dataset...
resize_y, resize_x = 6, 6
resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()
config_dataset.resize = resize
config_dataset.mask   = mask
dataset_validate = SiameseTestset(config_dataset)

# Obtain the new size...
spiimg = SPIImgDataset(config_dataset)
img    = spiimg.get_img_and_label(0)[0]
size_y, size_x = img.shape

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb = 10
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
alpha = 1.0
config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
model = SiameseModelCompare(config_siamese)

# Read chkpt from a trainig
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = 200,
                                    isflat      = False,
                                    max_epochs  = 1,    # Epoch = 1 for validate
                                    lr          = 0.001, )


validator = PairValidator(model, dataset_validate, config_validator)
validator.validate()
