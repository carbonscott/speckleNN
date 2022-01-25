#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
import numpy as np


logger = logging.getLogger(__name__)

class TrainerConfig:
    path_chkpt  = None
    num_workers = 4
    batch_size  = 64
    max_epochs  = 10
    lr          = 0.001
    debug       = False

    def __init__(self, **kwargs):
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)


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
        logger.info(f"Saving {self.config_train.path_chkpt}")
        torch.save(model.state_dict(), self.config_train.path_chkpt)


    def train(self):
        """ The training loop.  """

        # Load model and training configuration
        model, config_train = self.model, self.config_train
        model_raw           = model.module if hasattr(model, "module") else model
        optimizer           = model_raw.configure_optimizers(config_train)

        # Debug on???
        debug = self.config_train.debug

        # Train each epoch
        for epoch in tqdm.tqdm(range(config_train.max_epochs)):
            model.train()
            dataset_train = self.dataset_train
            loader_train = DataLoader( dataset_train, shuffle     = True, 
                                                      pin_memory  = True, 
                                                      batch_size  = config_train.batch_size,
                                                      num_workers = config_train.num_workers )
            losses = []

            # Train each batch
            batch = tqdm.tqdm(enumerate(loader_train), total = len(loader_train))
            for step_id, entry in batch:
                if debug: 
                    img_anchor, img_pos, img_neg, label_anchor, \
                    title_anchor, title_pos, title_neg = entry

                    for i in range(len(label_anchor)):
                        print(f"Processing {title_anchor[i]}, {title_pos[i]}, {title_neg[i]}...")
                else: 
                    img_anchor, img_pos, img_neg, label_anchor = entry

                img_anchor = img_anchor.to(self.device)
                img_pos    = img_pos.to(self.device)
                img_neg    = img_neg.to(self.device)

                ## print(f"{step_id:04d}, {img_anchor.shape}.")

                optimizer.zero_grad()

                _, _, _, loss = self.model.forward(img_anchor, img_pos, img_neg)
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().detach().numpy())

                print(f"Batch loss: {np.mean(loss.cpu().detach().numpy()):.4f}")

            print(f"Epoch: {epoch + 1}/{config_train.max_epochs} - Loss: {np.mean(loss.cpu().detach().numpy()):.4f}")

            # Save the model state
            self.save_checkpoint()
