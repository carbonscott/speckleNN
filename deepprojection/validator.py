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
            logger.info(f"KV - {k:16s} : {v}")




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


    def validate(self, is_return_loss = False, epoch = None):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Validate an epoch...
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

            loss_batch_mean = np.mean(losses_batch)
            logger.info(f"MSG - epoch {epoch}, batch {step_id:d}, loss {loss_batch_mean:.4f}")
            losses_epoch.append(loss_batch_mean)

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.4f}")

        return loss_epoch_mean if is_return_loss else None




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

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
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




class MultiwayQueryValidator:
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

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
        for step_id, entry in batch:
            # Unpack entry...
            # Not a good design, but it's okay for now (03/03/2020)
            # Shape of entry: (unpack_dim, batch, internal_shape)
            # Internal shape of image is 2d, string is 1d
            img_list   = entry[                 : len(entry) // 2 ]
            title_list = entry[ len(entry) // 2 :                 ]

            # Assign returned subcategory for img and title...
            img_query  , imgs_test   = img_list[0]  , img_list[1:]
            title_query, titles_test = title_list[0], title_list[1:]

            # Load imgs to gpu...
            img_query = img_query.to(self.device)
            imgs_test = [ img_test.to(self.device) for img_test in imgs_test ]

            with torch.no_grad():
                # Look at each example in a batch...
                for i in range(len(img_query)):
                    # Compare the query against EACH test image in the subcategory (each label)...
                    msg = []
                    for img_test, title_test in zip(imgs_test, titles_test):
                        _, _, dist = self.model.forward(img_query[i].unsqueeze(0), img_test[i].unsqueeze(0))

                        dist_val = dist.cpu().detach().numpy()
                        msg.append(f"{title_test[i]} : {dist_val:7.4f}")


                    # Return a line for each batch...
                    log_header = f"DATA - {title_query[i]}, "
                    log_msg = log_header + ", ".join(msg)
                    logger.info(log_msg)




class SimpleEmbeddingChecker:
    def __init__(self, model, dataset_test, config_test):
        self.model        = model
        self.dataset_test = dataset_test
        self.config_test  = config_test

        # Load data to gpus if available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

            if self.config_test.path_chkpt is not None:
                chkpt = torch.load(self.config_test.path_chkpt)
                self.model.load_state_dict(chkpt)
            else:
                # Initialize weights...
                def init_weights(module):
                    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                        module.weight.data.normal_(mean = 0.0, std = 0.02)
                self.model.apply(init_weights)
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        return None


    def run(self):
        """ The testing loop.  """

        # Load model and testing configuration...
        model, config_test = self.model, self.config_test

        # Train an epoch...
        # Load model state
        model.eval()
        dataset_test = self.dataset_test
        loader_test  = DataLoader( dataset_test, shuffle     = config_test.shuffle, 
                                                 pin_memory  = config_test.pin_memory, 
                                                 batch_size  = config_test.batch_size,
                                                 num_workers = config_test.num_workers )

        # Train each batch...
        batch = tqdm.tqdm(enumerate(loader_test), total = len(loader_test))
        for batch_id, entry in batch:
            # Unpack entry...
            # Not a good design, but it's okay for now (03/03/2020)
            # Shape of entry: (unpack_dim, batch, internal_shape)
            # Internal shape of image is 2d, string is 1d
            img_single = entry[0]
            title      = entry[1]

            # Load imgs to gpu...
            img_single = img_single.to(self.device)

            with torch.no_grad():
                # Look at each example in a batch...
                for i in range(len(img_single)):
                    # Inference...
                    img_embed = self.model.forward(img_single[i].unsqueeze(0))

                    # Fetch the descriptor...
                    msg = title[i]

                    # Return a line for each batch...
                    log_header = f"DATA - {batch_id * len(img_single) + i:06d} - "
                    log_msg = log_header + msg
                    logger.info(log_msg)

                    # Save the embedding...
                    if batch_id + i == 0:
                        size_y, size_x = img_embed.shape
                        num_imgs       = len(img_single) * len(batch)
                        imgs = torch.zeros(num_imgs, size_y, size_x)
                    imgs[i + batch_id * len(img_single)] = img_embed

        return imgs
