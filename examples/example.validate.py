#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig( format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )

import torch
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import SiameseModel, SiameseConfig
from deepprojection.validator import ValidatorConfig, Validator
import os

fl_csv = 'datasets.csv'
size_sample = 20
debug = True
dataset_test = SiameseDataset(fl_csv, size_sample, debug = debug)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)

config_siamese = SiameseConfig(alpha = 1.0, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)

drc_cwd = os.getcwd()
path_chkpt = os.path.join(drc_cwd, "trained_model.chkpt")
config_test = ValidatorConfig( path_chkpt  = path_chkpt,
                               num_workers   = 1,
                               batch_size    = 20,
                               max_epochs    = 1,
                               lr            = 0.001, 
                               debug         = debug, )

tester = Validator(model, dataset_test, config_test)
tester.test()
