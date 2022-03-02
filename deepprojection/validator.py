#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class ConfigValidator:
    path_chkpt   = None
    num_workers  = 4
    batch_size   = 64
    max_epochs   = 10
    lr           = 0.001
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"__/ Configure Validator \___")
        # Set values of attributes that are not known when obj is created...
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"{k:16s} : {v}")




class LossValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available...
        self.device = 'cpu'
        if self.config_test.path_chkpt is not None and torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def validate(self, is_return_loss = False, main_epoch = None):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train each epoch...
        for epoch in tqdm.tqdm(range(config_test.max_epochs)):
            epoch_str = f"{main_epoch:d}:{epoch:d}" if main_epoch is not None else "{epoch:d}"

            # Load model state...
            model.eval()
            dataset_test = self.dataset_test
            loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                     pin_memory  = config_test.pin_memory, 
                                                     batch_size  = config_test.batch_size,
                                                     num_workers = config_test.num_workers )

            # Train each batch...
            losses_epoch = []
            batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test), disable = config_test.tqdm_disable)
            for step_id, entry in batch:
                losses_batch = []

                img_anchor, img_pos, img_neg, label_anchor, \
                title_anchor, title_pos, title_neg = entry

                img_anchor = img_anchor.to(self.device)
                img_pos    = img_pos.to(self.device)
                img_neg    = img_neg.to(self.device)

                for i in range(len(label_anchor)):
                    logger.info(f"DATA - {title_anchor[i]}, {title_pos[i]}, {title_neg[i]}")

                with torch.no_grad():
                    _, _, _, loss = self.model.forward(img_anchor, img_pos, img_neg)
                    loss_val = loss.cpu().detach().numpy()
                    losses_batch.append(loss_val)
                    logger.info(f"DATA - {title_anchor[i]}, {title_pos[i]}, {title_neg[i]}, {loss_val:7.4f}")

                loss_batch_mean = np.mean(losses_batch)
                logger.info(f"MSG - epoch {epoch_str}, batch {step_id:d}, loss {loss_batch_mean:.4f}")
                losses_epoch.append(loss_batch_mean)

            loss_epoch_mean = np.mean(losses_epoch)
            logger.info(f"MSG - epoch {epoch_str}, loss mean {loss_epoch_mean:.4f}")

        # Record the mean loss of the last epoch...
        final_loss = loss_epoch_mean

        return final_loss if is_return_loss else None




class PairValidator:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            chkpt = torch.load(self.config_test.path_chkpt)
            self.model.load_state_dict(chkpt)
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
            loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                     pin_memory  = config_test.pin_memory, 
                                                     batch_size  = config_test.batch_size,
                                                     num_workers = config_test.num_workers )

            # Debug purpose
            self.loader_test = loader_test

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
                            ## print(title_anchor[i], title_second[i])
                            _, _, rmsd = self.model.forward(img_anchor[i].unsqueeze(0), img_second[i].unsqueeze(0))

                        rmsd_val = rmsd.cpu().detach().numpy()
                        rmsds.append(rmsd_val)
                        logger.info(f"DATA - {title_anchor[i]}, {title_second[i]}, {rmsd_val:7.4f}")
