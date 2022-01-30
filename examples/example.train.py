#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.experiments import SPIImgDataset, SiameseDataset
from deepprojection.model       import SiameseModel, SiameseConfig
from deepprojection.trainer     import TrainerConfig, Trainer

logging.basicConfig( filename = f"{__file__[:__file__.rfind('.py')]}.log",
                     filemode = 'w',
                     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        ## print(module)
        module.weight.data.normal_(mean = 0.0, std = 0.02)

fl_csv = 'datasets.csv'
size_sample = 1000
debug = True
## dataset_train = SiameseDataset(fl_csv, size_sample, debug = debug)

## resize_y, resize_x = 128, 128
## dataset_train = SiameseDataset(fl_csv, size_sample, resize = (resize_y, resize_x), debug = debug)
dataset_train = SiameseDataset(fl_csv, size_sample, resize = (), debug = debug)

# Get image size
## spiimg = SPIImgDataset(fl_csv, resize = (resize_y, resize_x))
spiimg = SPIImgDataset(fl_csv, resize = ())
size_y, size_x = spiimg.get_imagesize(0)

## # Get image size
## spiimg = SPIImgDataset(fl_csv)
## size_y, size_x = spiimg.get_imagesize(0)

# Try different margin (alpha) for Siamese net
for i, alpha in enumerate((1.0, )):
    config_siamese = SiameseConfig(alpha = alpha, dim_emb = int(size_y / 4), size_y = size_y, size_x = size_x)
    model = SiameseModel(config_siamese)
    model.apply(init_weights)

    drc_cwd = os.getcwd()
    path_chkpt = os.path.join(drc_cwd, f"trained_model.{i:02d}.chkpt")
    logger.info(f"alpha = {alpha}, checkpoint = trained_model.{i:02d}.chkpt.")
    config_train = TrainerConfig( path_chkpt  = path_chkpt,
                                  num_workers = 1,
                                  batch_size  = 100,
                                  max_epochs  = 10,
                                  lr          = 0.001, 
                                  debug       = debug, )

    trainer = Trainer(model, dataset_train, config_train)
    trainer.train()
