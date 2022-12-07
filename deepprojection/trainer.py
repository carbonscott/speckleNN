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
    def __init__(self, model, dataset, config):
        self.model   = model
        self.dataset = dataset
        self.config  = config

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def save_checkpoint(self):
        # Hmmm, DataParallel wrappers keep raw model object in .module attribute
        model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"SAVE - {self.config.path_chkpt}")
        torch.save(model.state_dict(), self.config.path_chkpt)


    def train(self, saves_checkpoint = True, epoch = None, logs_batch_loss = False):
        """ The training loop.  """

        # Load model and training configuration...
        # Optimizer can be reconfigured next epoch
        model, config = self.model, self.config
        model_raw     = model.module if hasattr(model, "module") else model
        optimizer     = model_raw.configure_optimizers(config)

        # Train an epoch...
        model.train()
        dataset = self.dataset
        loader_train = DataLoader( dataset, shuffle     = config.shuffle, 
                                                  pin_memory  = config.pin_memory, 
                                                  batch_size  = config.batch_size,
                                                  num_workers = config.num_workers )
        losses_epoch = []

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config.tqdm_disable)
        for step_id, entry in batch:
            img_anchor, img_pos, img_neg, label_anchor, \
            metadata_anchor, metadata_pos, metadata_neg = entry

            for i in range(len(label_anchor)):
                logger.info(f"DATA - {metadata_anchor[i]}, {metadata_pos[i]}, {metadata_neg[i]}")

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
    def __init__(self, model, dataset, config):
        self.model         = model
        self.dataset = dataset
        self.config  = config

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device, dtype = torch.float)

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
        model, config = self.model, self.config
        model_raw     = model.module if hasattr(model, "module") else model
        optimizer     = model_raw.configure_optimizers(config)

        # Train an epoch...
        model.train()
        dataset = self.dataset
        loader_train = DataLoader( dataset, shuffle     = config.shuffle, 
                                                  pin_memory  = config.pin_memory, 
                                                  batch_size  = config.batch_size,
                                                  num_workers = config.num_workers )
        losses_epoch = []

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config.tqdm_disable)
        for step_id, entry in batch:
            batch_imgs, batch_labels, batch_metadata = entry
            batch_imgs = batch_imgs.to(self.device, dtype = torch.float)

            loss = self.model.forward(batch_imgs, batch_labels, batch_metadata, 
                                      logs_triplets = config.logs_triplets, 
                                      method        = config.method,)

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




class SimpleTrainer:
    def __init__(self, model, dataset, config):
        self.model         = model
        self.dataset = dataset
        self.config  = config

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model  = torch.nn.DataParallel(self.model).to(self.device, dtype = torch.float)

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
        model, config = self.model, self.config
        model_raw     = model.module if hasattr(model, "module") else model
        optimizer     = model_raw.configure_optimizers(config)

        # Train an epoch...
        model.train()
        dataset = self.dataset
        loader_train = DataLoader( dataset, shuffle     = config.shuffle, 
                                            pin_memory  = config.pin_memory, 
                                            batch_size  = config.batch_size,
                                            num_workers = config.num_workers )
        losses_epoch = []

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config.tqdm_disable)
        for step_id, entry in batch:
            batch_imgs, batch_labels, batch_metadata = entry
            batch_imgs = batch_imgs.to(self.device, dtype = torch.float)
            batch_labels = batch_labels[:, None].to(self.device, dtype = torch.float)

            _, loss = self.model.forward(batch_imgs, batch_labels)

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
