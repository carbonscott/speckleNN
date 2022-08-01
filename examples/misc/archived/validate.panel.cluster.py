#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.panels  import SimpleSet             , ConfigDataset
from deepprojection.model            import SimpleEmbedding       , ConfigSiameseModel
from deepprojection.validator        import SimpleEmbeddingChecker, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122          , ConfigEncoder
from deepprojection.datasets         import transform
from deepprojection.utils            import MetaLog
from panel_preprocess                import DatasetPreprocess
import socket

# Comment this verification...
hostname = socket.gethostname()
comments = f""" 
            Hostname: {hostname}.
            Study if crop with a bin of 6 could beat the performance of the same setting but without cropping.  
            resize = 6
            Standardization is applied.  
            size_sample = 2000.
            alpha       = 2.
            cropping is applied.  
            """

# Create a timestamp to name the log file...
timestamp = "20220316134804"

# Validate mode...
istrain = False
mode_validate = 'train' if istrain else 'test'

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
basename       = f"{timestamp}.validate.simple.{mode_validate}.raw"
fl_log         = f"{basename}.log"
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

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP ]
panels         = [ 0, 1 ,2 ]    # Exclude panel 3 that has sick panel from run 102
config_dataset = ConfigDataset( fl_csv            = 'datasets.csv',
                                size_sample       = 2000, 
                                mode              = 'calib',
                                mask              = None,
                                resize            = None,
                                seed              = 0,
                                panels            = panels,
                                isflat            = False,
                                istrain           = istrain,
                                trans_random      = None,
                                trans_standardize = None,
                                frac_train        = 0.7,
                                exclude_labels    = exclude_labels, )

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset_preproc = DatasetPreprocess(config_dataset)
dataset_preproc.apply()
size_y, size_x = dataset_preproc.get_panelsize()

# Define validation set...
config_dataset.report()
dataset_validate = SimpleSet(config_dataset)

DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

# Config the encoder...
dim_emb = 128
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
config_siamese = ConfigSiameseModel( encoder = encoder, )
model = SimpleEmbedding(config_siamese)

# Read chkpt from a trainig
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( path_chkpt  = None,
                                    ## path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = 40,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = 1e-3, )

validator = SimpleEmbeddingChecker(model, dataset_validate, config_validator)
imgs = validator.run()

# Fetch embedding directory...
fl_emb = f"{basename}.pt"
DRCEMB = "embeds"
prefixpath_emb = os.path.join(drc_cwd, DRCEMB)
if not os.path.exists(prefixpath_emb): os.makedirs(prefixpath_emb)
path_emb = os.path.join(prefixpath_emb, fl_emb)

torch.save(imgs, path_emb)
