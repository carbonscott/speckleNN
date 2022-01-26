#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig( filename = f"{__file__[:__file__.rfind('.py')]}.log",
                     filemode = 'w',
                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)


import torch
from deepprojection.datasets.simulated import SiameseDataset
from deepprojection.model              import SiameseModel, SiameseConfig
from deepprojection.validator          import ValidatorConfig, Validator
import os

size_sample = 1000
debug = True
dataset_test = SiameseDataset(size_sample, debug = debug)

# Get image size
size_y, size_x = 128, 128

config_siamese = SiameseConfig(alpha = 10.0, size_y = size_y, size_x = size_x)
model = SiameseModel(config_siamese)

drc_cwd = os.getcwd()
path_chkpt = os.path.join(drc_cwd, "simulated.trained_model.00.chkpt")
config_test = ValidatorConfig( path_chkpt  = path_chkpt,
                               num_workers = 1,
                               batch_size  = 200,
                               max_epochs  = 10,
                               lr          = 0.001, 
                               debug       = debug, )

tester = Validator(model, dataset_test, config_test)
tester.test()
