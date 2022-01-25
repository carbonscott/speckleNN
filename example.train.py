#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from deepprojection.dataset import SPIImgDataset, SiameseDataset
from deepprojection.model   import SiameseModel, SiameseConfig
from deepprojection.trainer import TrainerConfig, Trainer

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
size_sample = 500
debug = True
dataset_train = SiameseDataset(fl_csv, size_sample, debug = debug)

# Get image size
spiimg = SPIImgDataset(fl_csv)
size_y, size_x = spiimg.get_imagesize(0)

# Try different margin (alpha) for Siamese net
for i, alpha in enumerate((10.0, 5.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)):
    config_siamese = SiameseConfig(alpha = alpha, size_y = size_y, size_x = size_x)
    model = SiameseModel(config_siamese)
    model.apply(init_weights)

    drc_cwd = os.getcwd()
    path_chkpt = os.path.join(drc_cwd, f"trained_model.{i:02d}.chkpt")
    logger.info(f"alpha = {alpha}, checkpoint = trained_model.{i:02d}.chkpt.")
    config_train = TrainerConfig( path_chkpt  = path_chkpt,
                                  num_workers = 1,
                                  batch_size  = 100,
                                  max_epochs  = 4,
                                  lr          = 0.001, 
                                  debug       = debug, )

    trainer = Trainer(model, dataset_train, config_train)
    trainer.train()
