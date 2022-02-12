#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class ConfigValidator:
    path_chkpt  = None
    num_workers = 4
    batch_size  = 64
    max_epochs  = 10
    lr          = 0.001

    def __init__(self, **kwargs):
        logger.info(f"__/ Configure Validator \___")
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"{k:16s} : {v}")




class LossValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            self.model.load_state_dict(torch.load(self.config_test.path_chkpt))
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self):
        """ The testing loop.  """

        # Load model and testing configuration
        model, config_test = self.model, self.config_test

        # Train each epoch
        for epoch in tqdm.tqdm(range(config_test.max_epochs)):
            # Load model state
            model.eval()
            dataset_test = self.dataset_test
            loader_test  = DataLoader( dataset_test, shuffle     = True, 
                                                     pin_memory  = True, 
                                                     batch_size  = config_test.batch_size,
                                                     num_workers = config_test.num_workers )

            # Train each batch
            batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
            for step_id, entry in batch:
                losses = []

                img_anchor, img_pos, img_neg, label_anchor, \
                title_anchor, title_pos, title_neg = entry

                img_anchor = img_anchor.to(self.device)
                img_pos    = img_pos.to(self.device)
                img_neg    = img_neg.to(self.device)

                with torch.no_grad():
                    # Look at each example in a batch...
                    for i in range(len(label_anchor)):
                        _, _, _, loss = self.model.forward(img_anchor[i], img_pos[i], img_neg[i])
                        loss_val = loss.cpu().detach().numpy()
                        losses.append(loss_val)
                        logger.info(f"DATA - {title_anchor[i]}, {title_pos[i]}, {title_neg[i]}, {loss_val:7.4f}")

                logger.info(f"MSG - epoch {epoch:d}, batch {step_id:d}, loss {np.mean(losses):.4f}")




class PairValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            self.model.load_state_dict(torch.load(self.config_test.path_chkpt))
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self):
        """ The testing loop.  """

        # Load model and testing configuration
        model, config_test = self.model, self.config_test

        # Train each epoch
        for epoch in tqdm.tqdm(range(config_test.max_epochs)):
            # Load model state
            model.eval()
            dataset_test = self.dataset_test
            loader_test  = DataLoader( dataset_test, shuffle     = True, 
                                                     pin_memory  = True, 
                                                     batch_size  = config_test.batch_size,
                                                     num_workers = config_test.num_workers )

            # Train each batch
            batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
            for step_id, entry in batch:
                rmsds = []

                img_anchor, img_second, label_anchor, title_anchor, title_second = entry

                img_anchor = img_anchor.to(self.device)
                img_second = img_second.to(self.device)

                with torch.no_grad():
                    # Look at each example in a batch...
                    for i in range(len(label_anchor)):
                        # Biolerplate unsqueeze due to the design of PyTorch Conv2d, no idea how to improve yet...
                        if config_test.isflat:
                            _, _, rmsd = self.model.forward(img_anchor[i], img_second[i])
                        else:
                            _, _, rmsd = self.model.forward(img_anchor[i].unsqueeze(0), img_second[i].unsqueeze(0))

                        rmsd_val = rmsd.cpu().detach().numpy()
                        rmsds.append(rmsd_val)
                        logger.info(f"DATA - {title_anchor[i]}, {title_second[i]}, {rmsd_val:7.4f}")
