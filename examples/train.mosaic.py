#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.mosaic  import SiameseDataset, ConfigDataset
from deepprojection.model            import SiameseModel  , ConfigSiameseModel
from deepprojection.trainer          import Trainer       , ConfigTrainer
from deepprojection.validator        import LossValidator , ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122  , ConfigEncoder
from deepprojection.utils            import EpochManager
from datetime import datetime
from mosaic_preprocess import DatasetPreprocess

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train.log"
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
panels_ordered = [1, 0]
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
config_dataset = ConfigDataset( fl_csv            = 'datasets.csv',
                                size_sample       = 2000, 
                                mode              = 'calib',
                                mask              = None,
                                resize            = None,
                                seed              = 0,
                                isflat            = False,
                                istrain           = True,
                                trans_random      = None,
                                trans_standardize = None,
                                trans_crop        = None,
                                panels_ordered    = panels_ordered,
                                frac_train        = 0.7,
                                exclude_labels    = exclude_labels, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset)
dataset_preproc.apply()
size_y, size_x = dataset_preproc.get_mosaicsize()

# Define training set...
config_dataset.report()
dataset_train = SiameseDataset(config_dataset)

# Define validation set...
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
alpha = 2.0
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
fl_chkpt         = f"{timestamp}.train.chkpt"
path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)


# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( path_chkpt  = path_chkpt,
                              num_workers = 1,
                              batch_size  = 40,
                              pin_memory  = True,
                              shuffle     = False,
                              lr          = 1e-3, )

# Training...
trainer = Trainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt  = None,
                                    num_workers = 1,
                                    batch_size  = 40,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    lr          = 1e-3, 
                                    isflat      = False, )  # Conv2d input needs one more dim for batch

validator = LossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 60
epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
epoch_manager.run()
