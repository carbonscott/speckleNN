#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class ConfigTrainer:
    path_chkpt   = None
    num_workers  = 4
    batch_size   = 64
    max_epochs   = 10
    lr           = 0.001
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Trainer \___")
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class Trainer:
    def __init__(self, model, dataset_train, config_train):
        self.model         = model
        self.dataset_train = dataset_train
        self.config_train  = config_train

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def save_checkpoint(self):
        # Hmmm, DataParallel wrappers keep raw model object in .module attribute
        model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"SAVE - {self.config_train.path_chkpt}")
        torch.save(model.state_dict(), self.config_train.path_chkpt)


    def train(self, saves_checkpoint = True, epoch = None, logs_batch_loss = False):
        """ The training loop.  """

        # Load model and training configuration...
        # Optimizer can be reconfigured next epoch
        model, config_train = self.model, self.config_train
        model_raw           = model.module if hasattr(model, "module") else model
        optimizer           = model_raw.configure_optimizers(config_train)

        # Train an epoch...
        model.train()
        dataset_train = self.dataset_train
        loader_train = DataLoader( dataset_train, shuffle     = config_train.shuffle, 
                                                  pin_memory  = config_train.pin_memory, 
                                                  batch_size  = config_train.batch_size,
                                                  num_workers = config_train.num_workers )
        losses_epoch = []

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config_train.tqdm_disable)
        for step_id, entry in batch:
            img_anchor, img_pos, img_neg, label_anchor, \
            title_anchor, title_pos, title_neg = entry

            for i in range(len(label_anchor)):
                logger.info(f"DATA - {title_anchor[i]}, {title_pos[i]}, {title_neg[i]}")

            img_anchor = img_anchor.to(self.device)
            img_pos    = img_pos.to(self.device)
            img_neg    = img_neg.to(self.device)

            _, _, _, loss = self.model.forward(img_anchor, img_pos, img_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.cpu().detach().numpy()
            losses_epoch.append(loss_val)

            if logs_batch_loss: 
                logger.info(f"MSG - epoch {epoch}, batch {step_id:d}, loss {loss_val:.8f}")

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        # Save the model state
        if saves_checkpoint: self.save_checkpoint()




class OnlineTrainer:
    def __init__(self, model, dataset_train, config_train):
        self.model         = model
        self.dataset_train = dataset_train
        self.config_train  = config_train

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def save_checkpoint(self, timestamp):
        DRCCHKPT = "chkpts"
        drc_cwd = os.getcwd()
        prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
        if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
        fl_chkpt   = f"{timestamp}.train.chkpt"
        path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)

        # Hmmm, DataParallel wrappers keep raw model object in .module attribute
        model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"SAVE - {path_chkpt}")
        torch.save(model.state_dict(), path_chkpt)


    def train(self, saves_checkpoint = True, epoch = None, returns_loss = False, logs_batch_loss = False):
        """ The training loop.  """

        # Load model and training configuration...
        # Optimizer can be reconfigured next epoch
        model, config_train = self.model, self.config_train
        model_raw           = model.module if hasattr(model, "module") else model
        optimizer           = model_raw.configure_optimizers(config_train)

        # Train an epoch...
        model.train()
        dataset_train = self.dataset_train
        loader_train = DataLoader( dataset_train, shuffle     = config_train.shuffle, 
                                                  pin_memory  = config_train.pin_memory, 
                                                  batch_size  = config_train.batch_size,
                                                  num_workers = config_train.num_workers )
        losses_epoch = []

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config_train.tqdm_disable)
        for step_id, entry in batch:
            batch_imgs, batch_labels, batch_titles = entry
            batch_imgs = batch_imgs.to(self.device)

            loss = self.model.forward(batch_imgs, batch_labels, batch_titles, 
                                      is_logging = config_train.is_logging, 
                                      method     = config_train.method,
                                      shuffle    = config_train.online_shuffle,)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.cpu().detach().numpy()
            losses_epoch.append(loss_val)

            if logs_batch_loss:
                logger.info(f"MSG - epoch {epoch}, batch {step_id:d}, loss {loss_val:.8f}")

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if returns_loss else None
