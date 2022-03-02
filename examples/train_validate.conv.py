#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch
from deepprojection.datasets.experiments import SPIImgDataset, SiameseDataset, ConfigDataset
from deepprojection.model                import SiameseModel , ConfigSiameseModel
from deepprojection.trainer              import Trainer      , ConfigTrainer
from deepprojection.validator            import LossValidator, ConfigValidator
from deepprojection.encoders.convnet     import Hirotaka0122 , ConfigEncoder
from deepprojection.datasets             import transform
from deepprojection.utils                import EpochManager

from datetime import datetime

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train_validate.log"
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

# [[[ DATASET ]]]
# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
config_dataset = ConfigDataset( fl_csv         = 'datasets.csv',
                                size_sample    = 4000, 
                                mode           = 'image',
                                mask           = None,
                                resize         = None,
                                seed           = 0,
                                isflat         = False,
                                istrain        = True,
                                trans          = None,
                                frac_train     = 0.7,
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

# Random transformation for data augmentation...
# Random rotation
angle = None
center = (524, 506)
trans_random_rotate = transform.RandomRotate(angle = angle, center = center)

# Random patching
num_patch = 5
size_patch_y, size_patch_x = 70, 500
trans_random_patch  = transform.RandomPatch(num_patch             , size_patch_y     , size_patch_x, 
                                            var_patch_y    = 0.2  , var_patch_x    = 0.2, 
                                            is_return_mask = False, is_random_flip = True)
trans_list = [trans_random_rotate, trans_random_patch]
## config_dataset.trans = trans_list

# Reconfig the dataset...
resize_y, resize_x = 6, 6
resize = (resize_y, resize_x) if not None in (resize_y, resize_x) else ()
config_dataset.resize = resize
config_dataset.mask = mask
config_dataset.report()
dataset_train = SiameseDataset(config_dataset)

# Obtain the new size...
spiimg = SPIImgDataset(config_dataset)
img    = spiimg.get_img_and_label(0)[0]
size_y, size_x = img.shape

# Config the validate set...
config_dataset.istrain = False
config_dataset.report()
dataset_validate = SiameseDataset(config_dataset)

# [[[ IMAGE ENCODER ]]]
# Config the encoder...
dim_emb = 128
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# [[[ MODEL ]]]
# Config the model...
alpha = 2.6
config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
model = SiameseModel(config_siamese)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
model.apply(init_weights)

# [[[ CHECKPOINT ]]]
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
fl_chkpt         = f"{timestamp}.train_validate.chkpt"
path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)

# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( path_chkpt  = path_chkpt,
                              num_workers = 1,
                              batch_size  = 40,
                              pin_memory  = True,
                              shuffle     = False,
                              max_epochs  = 1,
                              lr          = 1e-3, )

# Training...
trainer = Trainer(model, dataset_train, config_train)

# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt  = None,
                                    num_workers = 1,
                                    batch_size  = 40,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    max_epochs  = 1,    # Epoch = 1 for validate
                                    lr          = 1e-3, )

validator = LossValidator(model, dataset_validate, config_validator)

# [[[ EPOCH MANAGER ]]]
max_epochs = 40
epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
epoch_manager.run()
