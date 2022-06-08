#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.datasets.mosaic  import SequentialSet      , ConfigDataset
from deepprojection.model            import EmbeddingModel     , ConfigSiameseModel
from deepprojection.validator        import EmbeddingCalculator, ConfigValidator
from deepprojection.encoders.convnet import Hirotaka0122       , ConfigEncoder
from deepprojection.datasets         import transform
from deepprojection.utils            import MetaLog
from mosaic_preprocess               import DatasetPreprocess
import socket

# Create a timestamp to name the log file...
timestamp = "2022_0603_2226_44"

# Set up parameters for an experiment...
fl_csv         = 'datasets.precomp_emb.csv'
size_sample    = None
size_batch     = 40
online_shuffle = True
lr             = 1e-3
frac_train     = 1.0
seed           = 0
istrain        = True
panels_ordered = [1, 2]

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size    : {size_sample}
            Batch  size    : {size_batch}
            Online shuffle : {online_shuffle}
            lr             : {lr}

            """


# ___/ EMBEDDING CALCULATION \___
# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
basename       = f"{timestamp}.precomp_emb"
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
config_dataset = ConfigDataset( fl_csv         = fl_csv,
                                size_sample    = size_sample, 
                                psana_mode     = 'calib',
                                seed           = seed,
                                isflat         = False,
                                istrain        = istrain,
                                trans          = None,
                                frac_train     = frac_train,
                                panels_ordered = panels_ordered,
                                exclude_labels = exclude_labels, )

# Define validation set...
dataset = SequentialSet(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
dataset.MOSAIC_ON = False
img_orig, _       = dataset.get_img_and_label(0)
panel_orig        = img_orig[0]
dataset.MOSAIC_ON = True
dataset_preproc   = DatasetPreprocess(panel_orig)
trans             = dataset_preproc.config_trans()
dataset.trans     = trans
mosaic_trans, _   = dataset.get_img_and_label(0)

# Report training set...
config_dataset.trans = trans
config_dataset.report()

# Config the encoder...
dim_emb        = 128
size_y, size_x = mosaic_trans.shape
config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                size_y  = size_y,
                                size_x  = size_x,
                                isbias  = True )
encoder = Hirotaka0122(config_encoder)

# Set up the model
config_siamese = ConfigSiameseModel( encoder = encoder, )
model = EmbeddingModel(config_siamese)

# Read chkpt from a trainig
DRCCHKPT = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
fl_chkpt = f"{timestamp}.train.chkpt"
path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
config_validator = ConfigValidator( ## path_chkpt  = None,    # Use it for raw input (random weights)
                                    path_chkpt  = path_chkpt,
                                    num_workers = 1,
                                    batch_size  = size_batch,
                                    pin_memory  = True,
                                    shuffle     = False,
                                    isflat      = False,
                                    lr          = lr, )
validator = EmbeddingCalculator(model, dataset, config_validator)
embs = validator.run()

# Fetch embedding directory...
fl_emb = f"{basename}.pt"
DRCEMB = "embeds"
prefixpath_emb = os.path.join(drc_cwd, DRCEMB)
if not os.path.exists(prefixpath_emb): os.makedirs(prefixpath_emb)
path_emb = os.path.join(prefixpath_emb, fl_emb)

fl_emb_avg_dict = f"{basename}.emb_avg_dict.pt"
path_emb_avg_dict = os.path.join(prefixpath_emb, fl_emb_avg_dict)

fl_label_seqi_orig_dict = f"{basename}.label_seqi_orig_dict.pt"
path_label_seqi_orig_dict = os.path.join(prefixpath_emb, fl_label_seqi_orig_dict)

# ___/ AVERAGING EMBEDDING PER CLASS \___
emb_avg_dict = {}
for k, v in dataset.label_seqi_orig_dict.items():
    emb_avg_dict[k] = embs[v].mean(dim = 0)    # Mean along sample dim

torch.save(embs                        , path_emb                 )
torch.save(emb_avg_dict                , path_emb_avg_dict        )
torch.save(dataset.label_seqi_orig_dict, path_label_seqi_orig_dict)
