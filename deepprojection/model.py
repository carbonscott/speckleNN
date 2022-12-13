#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import random
from deepprojection.utils import calc_dmat, set_seed
from itertools import combinations, permutations

import logging

logger = logging.getLogger(__name__)

class ConfigSiameseModel:
    alpha   = 0.5

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Siamese Model \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16} : {v}")




class SiameseModel(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.alpha   = getattr(config, "alpha"  , None)
        self.encoder = getattr(config, "encoder", None)
        self.seed    = getattr(config, "seed"   , None)

        if self.seed is not None:
            set_seed(self.seed)


    def init_params(self, from_timestamp = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if from_timestamp is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            fl_chkpt_prev    = f"{from_timestamp}.train.chkpt"
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt_prev)
            self.load_state_dict(torch.load(path_chkpt_prev))


    def forward(self, batch_anchor, batch_pos, batch_neg):
        # Encode images...
        batch_anchor_embed = self.encoder.encode(batch_anchor)
        batch_pos_embed    = self.encoder.encode(batch_pos)
        batch_neg_embed    = self.encoder.encode(batch_neg)

        # Calculate the RMSD between anchor and positive...
        # Well, it's in fact squared distance
        batch_diff = batch_anchor_embed - batch_pos_embed
        rmsd_anchor_pos = torch.sum(batch_diff * batch_diff, dim = -1)

        # Calculate the RMSD between anchor and negative...
        batch_diff = batch_anchor_embed - batch_neg_embed
        rmsd_anchor_neg = torch.sum(batch_diff * batch_diff, dim = -1)

        # Calculate the triplet loss, relu is another implementation of max(a, b)...
        loss_triplet = torch.relu(rmsd_anchor_pos - rmsd_anchor_neg + self.alpha)

        return loss_triplet.mean()


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr = config_train.lr)

        return optimizer




class OnlineSiameseModel(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.alpha   = getattr(config, "alpha"  , None)
        self.encoder = getattr(config, "encoder", None)
        self.seed    = getattr(config, "seed"   , None)

        if self.seed is not None:
            set_seed(self.seed)


    def init_params(self, from_timestamp = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if from_timestamp is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            fl_chkpt_prev    = f"{from_timestamp}.train.chkpt"
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt_prev)
            self.load_state_dict(torch.load(path_chkpt_prev))


    def forward(self, batch_imgs, batch_labels, batch_metadata, 
                      logs_triplets     = True, 
                      method            = 'semi-hard',):
        # Supposed methods
        select_method_dict = {
            'random-semi-hard' : self.batch_random_semi_hard,
            'random'           : self.batch_random,
        }

        # Assert a valid method...
        assert method in select_method_dict, f"{method} is not supported!!!"

        # Select triplets...
        select_method = select_method_dict[method]
        triplets = select_method(batch_imgs, batch_labels, batch_metadata, logs_triplets = logs_triplets)

        img_anchor = batch_imgs[ [ triplet[0] for triplet in triplets ] ]
        img_pos    = batch_imgs[ [ triplet[1] for triplet in triplets ] ]
        img_neg    = batch_imgs[ [ triplet[2] for triplet in triplets ] ]

        # Encode images...
        img_anchor_embed = self.encoder.encode(img_anchor)
        img_pos_embed    = self.encoder.encode(img_pos)
        img_neg_embed    = self.encoder.encode(img_neg)

        # Calculate the RMSD between anchor and positive...
        # Well, it's in fact squared distance
        img_diff = img_anchor_embed - img_pos_embed
        rmsd_anchor_pos = torch.sum(img_diff * img_diff, dim = -1)

        # Calculate the RMSD between anchor and negative...
        img_diff = img_anchor_embed - img_neg_embed
        rmsd_anchor_neg = torch.sum(img_diff * img_diff, dim = -1)

        # Calculate the triplet loss, relu is another implementation of max(a, b)...
        loss_triplet = torch.relu(rmsd_anchor_pos - rmsd_anchor_neg + self.alpha)

        return loss_triplet.mean()


    def batch_random(self, batch_imgs, batch_labels, batch_metadata, logs_triplets = True, **kwargs):
        ''' Totally random shuffled triplet.  
        '''
        # Convert batch labels to dictionary for fast lookup...
        batch_label_dict = {}
        batch_label_list = batch_labels.cpu().numpy()
        for i, v in enumerate(batch_label_list):
            if not v in batch_label_dict: batch_label_dict[v] = [i]
            else                        : batch_label_dict[v].append(i)

        # ___/ NEGATIVE MINIG \___
        # Go through each image in the batch and form triplets...
        # Prepare for logging
        triplets = []
        for batch_idx_achr, img in enumerate(batch_imgs):
            # Get the label of the image...
            batch_label_achr = batch_label_list[batch_idx_achr]

            # Create a bucket of positive cases...
            batch_idx_pos_list = batch_label_dict[batch_label_achr]

            # Select a positive case from positive bucket...
            batch_idx_pos = random.choice(batch_idx_pos_list)

            # Create a bucket of negative cases...
            idx_neg_list = []
            for batch_label, idx_list in batch_label_dict.items():
                if batch_label == batch_label_achr: continue
                idx_neg_list += idx_list

            # Randomly choose one negative example...
            idx_reduced   = random.choice(range(len(idx_neg_list)))
            batch_idx_neg = idx_neg_list[idx_reduced]

            # Track triplet...
            triplets.append((batch_idx_achr, batch_idx_pos, batch_idx_neg))

        if logs_triplets:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                metadata_achr = batch_metadata[batch_idx_achr]
                metadata_pos  = batch_metadata[batch_idx_pos]
                metadata_neg  = batch_metadata[batch_idx_neg]
                logger.info(f"DATA - {metadata_achr} {metadata_pos} {metadata_neg}")

        return triplets


    def batch_random_semi_hard(self, batch_imgs, batch_labels, batch_metadata, logs_triplets = True, **kwargs):
        ''' Only supply batch_size of image triplet for training.  This is in
            contrast to the all positive method.  Each image in the batch has
            the chance of playing an anchor.  Negative mining is applied.  
        '''
        # Retrieve the batch size...
        batch_size = batch_imgs.shape[0]

        # Encode all batched images without autograd tracking...
        with torch.set_grad_enabled(False):
            batch_embs = self.encoder.encode(batch_imgs)

        # Convert batch labels to dictionary for fast lookup...
        batch_label_dict = {}
        batch_label_list = batch_labels.cpu().numpy()
        for i, v in enumerate(batch_label_list):
            if not v in batch_label_dict: batch_label_dict[v] = [i]
            else                        : batch_label_dict[v].append(i)

        # ___/ NEGATIVE MINIG \___
        # Go through each image in the batch and form triplets...
        # Prepare for logging
        triplets = []
        dist_log = []
        for batch_idx_achr, img in enumerate(batch_imgs):
            # Get the label of the image...
            batch_label_achr = batch_label_list[batch_idx_achr]

            # Create a bucket of positive cases...
            batch_idx_pos_list = batch_label_dict[batch_label_achr]

            # Select a positive case from positive bucket...
            batch_idx_pos = random.choice(batch_idx_pos_list)

            # Find positive embedding squared distances..
            emb_achr = batch_embs[batch_idx_achr]
            emb_pos  = batch_embs[batch_idx_pos]
            delta_emb_pos = emb_achr - emb_pos
            dist_pos = torch.sum(delta_emb_pos * delta_emb_pos)

            # Create a bucket of negative cases...
            idx_neg_list = []
            for batch_label, idx_list in batch_label_dict.items():
                if batch_label == batch_label_achr: continue
                idx_neg_list += idx_list
            idx_neg_list = torch.tensor(idx_neg_list)

            # Collect all negative embeddings...
            emb_neg_list = batch_embs[idx_neg_list]

            # Find negative embedding squared distances...
            delta_emb_neg_list = emb_achr[None, :] - emb_neg_list
            dist_neg_list = torch.sum( delta_emb_neg_list * delta_emb_neg_list, dim = -1 )

            # Find negative squared distance satisfying dist_neg > dist_pos (semi hard)...
            # logical_and is only supported when pytorch version >= 1.5
            ## cond_semihard = torch.logical_and( dist_pos < dist_neg_list, dist_neg_list < dist_pos + self.alpha)
            cond_semihard = (dist_pos < dist_neg_list) * (dist_neg_list < dist_pos + self.alpha)

            # If semi hard exists???
            if torch.any(cond_semihard):
                ## # Look for the argmin...
                ## min_neg_semihard = torch.min(dist_neg_list[cond_semihard], dim = -1)
                ## batch_idx_neg    = idx_neg_list[cond_semihard][min_neg_semihard.indices]
                ## dist_neg = min_neg_semihard.values

                # Select one random example that is semi hard...
                size_semihard = torch.sum(cond_semihard)
                idx_random_semihard = random.choice(range(size_semihard))

                # Fetch the batch index of the example and its distance w.r.t the anchor...
                batch_idx_neg = idx_neg_list [cond_semihard][idx_random_semihard]
                dist_neg      = dist_neg_list[cond_semihard][idx_random_semihard]

            # Otherwise, randomly select one negative example???
            else:
                idx_reduced   = random.choice(range(len(idx_neg_list)))
                batch_idx_neg = idx_neg_list[idx_reduced]
                dist_neg      = dist_neg_list[idx_reduced]

            # Track triplet...
            triplets.append((batch_idx_achr, batch_idx_pos, batch_idx_neg.tolist()))
            dist_log.append((dist_pos, dist_neg))

        if logs_triplets:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                metadata_achr = batch_metadata[batch_idx_achr]
                metadata_pos  = batch_metadata[batch_idx_pos]
                metadata_neg  = batch_metadata[batch_idx_neg]
                dist_pos   = dist_log[idx][0]
                dist_neg   = dist_log[idx][1]
                logger.info(f"DATA - {metadata_achr} {metadata_pos} {metadata_neg} {dist_pos:12.6f} {dist_neg:12.6f}")

        return triplets


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr = config_train.lr)
        ## optimizer = torch.optim.SGD(self.encoder.parameters(), lr = config_train.lr)

        return optimizer




class SiameseModelCompare(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.encoder = config.encoder


    def init_params(self, from_timestamp = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if from_timestamp is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            fl_chkpt_prev    = f"{from_timestamp}.train.chkpt"
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt_prev)
            self.load_state_dict(torch.load(path_chkpt_prev))

        # Move to a device...
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self   = torch.nn.DataParallel(self).to(device)


    def forward(self, batch_img_query, batch_img_test):
        '''
        Parameters
        ----------
        batch_img_query : torch tensor, (num_query = 1     , size_batch, size_image2d)
        batch_img_test  : torch tensor, (num_test_per_query, size_batch, size_image2d)
        '''

        # Encode the query image...
        # batch_emb_query : (num_test_per_query, size_batch, len_emb)
        batch_emb_query = torch.stack( [ self.encoder.encode(each_img) for each_img in batch_img_query ] )

        # batch_emb_test  : (num_test_per_query, size_batch, len_emb)
        batch_emb_test = torch.stack( [ self.encoder.encode(each_img) for each_img in batch_img_test ] )

        # Calculate the RMSD between two embds -- query and test...
        # emb_diff     : (num_test_per_query, size_batch, len_emb)
        # dist_squared : (num_test_per_query, size_batch, )
        batch_emb_diff     = batch_emb_query - batch_emb_test
        batch_dist_squared = torch.sum(batch_emb_diff * batch_emb_diff, dim = -1)

        return batch_emb_query, batch_emb_test, batch_dist_squared




class EmbeddingModel(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.encoder = config.encoder


    def forward(self, img):
        return self.encoder.encode(img)




class Shi2019Model(nn.Module):
    """ DOI: 10.1107/S2052252519001854 """

    def __init__(self, config):
        super().__init__()
        self.encoder    = getattr(config, "encoder"   , None)
        self.seed       = getattr(config, "seed"      , None)
        self.pos_weight = getattr(config, "pos_weight", 1.0)

        self.pos_weight = torch.tensor(self.pos_weight, dtype = torch.float)

        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)

        if self.seed is not None:
            set_seed(self.seed)


    def init_params(self, from_timestamp = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if from_timestamp is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            fl_chkpt_prev    = f"{from_timestamp}.train.chkpt"
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt_prev)
            self.load_state_dict(torch.load(path_chkpt_prev))


    def forward(self, batch_img, batch_labels):
        # Encode images...
        batch_logit = self.encoder.forward(batch_img)

        # Calculate the BCE loss with logits...
        loss_bce = self.BCEWithLogitsLoss(batch_logit, batch_labels)

        return batch_logit, loss_bce


    def configure_optimizers(self, config):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr = config.lr)

        return optimizer

