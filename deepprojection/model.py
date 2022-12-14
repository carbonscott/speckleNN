#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import random
import numpy as np
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




class OnlineTripletSiameseModel(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.alpha   = getattr(config, "alpha"  , None)
        self.encoder = getattr(config, "encoder", None)
        self.seed    = getattr(config, "seed"   , None)

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

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


    def select_semi_hard(self, batch_encode, batch_candidate_list, batch_metadata_list = None, logs_triplets = False):
        """
        Return a list of entities of three elements -- `(a, p, n)`, which
        stand for anchor, positive and negative respectively.

        Parameters
        ----------
        batch_encode : tensor
            List of encoded labels packaged in Torch Tensor.
            Expected shape: [10]
            - num of sample in a mini-batch

        batch_candidate_list : tensor
            List of candidate images packaged in Torch Tensor.
            Expected shape: [ 10, 20, 1, 48, 48 ]
            - num of sample in a mini-batch
            - num of candidates
            - num of torch channel
            - size_y
            - size_x

        Returns
        -------
        triplet_list : tensor
            List of entities of three elements -- `(a, p, n)`, in which each
            elemnt is an image of shape `(1, size_y, size_x)`.
            Expected shape: [20, 1, 48, 48]
            - num of sample in a mini-batch
            - num of torch channel
            - size_y
            - size_x

        Examples
        --------
        >>> 

        """
        # Get the batch size...
        batch_size = len(batch_encode)

        # Get the shape of the batch_candidate_list...
        num_sample_in_mini_batch, num_candidate, size_c, size_y, size_x = batch_candidate_list.shape

        # Convert all input image into embeddings...
        with torch.no_grad():
            ###################################################################
            # These steps are here to facilitate the use of our PyTorch Model.
            ###################################################################
            # Compress sample dim and candidate dim into one example dim...
            batch_example_list = batch_candidate_list.view(-1, size_c, size_y, size_x)

            # Compute embeddings...
            batch_emb_list = self.encoder.encode(batch_example_list)

            # Reshape the first two dimension back to the original first two dimension...
            # Original dim: [num_sample_in_mini_batch, num_candidate, ...]
            batch_emb_list = batch_emb_list.view(num_sample_in_mini_batch, num_candidate, -1)

        # Build a lookup table to locate negative examples...
        encode_to_idx_dict = {}
        for idx_encode, encode in enumerate(batch_encode):
            encode = encode.item()
            if encode not in encode_to_idx_dict: encode_to_idx_dict[encode] = []
            encode_to_idx_dict[encode].append(idx_encode)

        # Go through each item in the mini-batch and find semi-hard triplets...
        triplet_list = []
        dist_list    = []
        for idx_encode, encode in enumerate(batch_encode):
            encode = encode.item()

            # Randomly choose an anchor and a positive from a candidate pool...
            idx_a, idx_p = random.sample(range(num_candidate), k = 2)
            emb_a = batch_emb_list[idx_encode][idx_a]
            emb_p = batch_emb_list[idx_encode][idx_p]

            # Calculate emb distance between a and p...
            # emb distance is defined as the squared L2
            diff_emb = emb_a - emb_p
            dist_p = torch.sum(diff_emb * diff_emb).item()

            # Fetch negative sample candidates...
            idx_encode_n_list = []
            for k_encode, v_idx_encode_list in encode_to_idx_dict.items():
                if k_encode == encode: continue
                idx_encode_n_list += v_idx_encode_list
            idx_encode_n_list = torch.tensor(idx_encode_n_list)

            # Collect all negative embeddings...
            emb_n_list = batch_emb_list[idx_encode_n_list]

            # Calculate emb distance between a and n...
            diff_emb_list = emb_a[None, :] - emb_n_list
            dist_n_list = torch.sum(diff_emb_list * diff_emb_list, dim = -1)

            # Create a logic expression to locate semi hard...
            cond_semihard = (dist_p < dist_n_list) * (dist_n_list < dist_p + self.alpha)

            # If semi hard exists???
            if torch.any(cond_semihard):
                # Selet idx_encode_n...
                # [IMPROVE] Higher version of PyTorch should make this process easier
                cond_semihard_numpy = cond_semihard.cpu().numpy()

                # Sample a semi hard example but represented using seqence id in cond_semihard_numpy...
                pos_semihard = np.argwhere(cond_semihard_numpy)
                seqi_encode, seqi_candidate = random.choice(pos_semihard)

                # Locate the semi hard using idx_encode and idx_candidate...
                idx_encode_n = idx_encode_n_list[seqi_encode].item()
                idx_n        = seqi_candidate

                # Record dist...
                dist_n = dist_n_list[seqi_encode][idx_n].item()

            # Otherwise, randomly select one negative example...
            else:
                seqi_encode  = random.choice(range(len(idx_encode_n_list)))
                idx_encode_n = idx_encode_n_list[seqi_encode].item()
                idx_n        = random.choice(range(dist_n_list.shape[-1]))
                dist_n       = dist_n_list[seqi_encode][idx_n].item()

            # Track variables for output...
            triplet_list.append(((idx_encode, idx_a), (idx_encode, idx_p), (idx_encode_n, idx_n)))
            dist_list.append((dist_p, dist_n))

        if logs_triplets:
            # Logging all cases...
            for idx, triplet in enumerate(triplet_list):
                (idx_encode, idx_a), (idx_encode, idx_p), (idx_encode_n, idx_n) = triplet

                metadata_a = batch_metadata_list[idx_encode  ][idx_a]
                metadata_p = batch_metadata_list[idx_encode  ][idx_p]
                metadata_n = batch_metadata_list[idx_encode_n][idx_n]

                annotate_semihard = ''
                dist_p, dist_n = dist_list[idx]
                diff_pn = dist_n - dist_p
                if 0 < diff_pn and diff_pn < self.alpha: annotate_semihard = f'semi-hard {diff_pn:e}'

                logger.info(f"DATA - {metadata_a:12s}, {metadata_p:12s}, {metadata_n:12s}; {annotate_semihard}")

        return triplet_list, dist_list


    def fetch_img_triplet(self, triplet_list, batch_candidate_list):
        # Get the shape of the batch_candidate_list...
        num_sample_in_mini_batch, num_candidate, size_c, size_y, size_x = batch_candidate_list.shape

        batch_candidate_flat_list = batch_candidate_list.view(num_sample_in_mini_batch * num_candidate, size_c, size_y, size_x)
        batch_a = batch_candidate_flat_list[ [ idx_encode * num_sample_in_mini_batch + idx_a for (idx_encode, idx_a), _, _ in triplet_list ] ]
        batch_p = batch_candidate_flat_list[ [ idx_encode * num_sample_in_mini_batch + idx_p for _, (idx_encode, idx_p), _ in triplet_list ] ]
        batch_n = batch_candidate_flat_list[ [ idx_encode * num_sample_in_mini_batch + idx_n for _, _, (idx_encode, idx_n) in triplet_list ] ]

        return batch_a, batch_p, batch_n


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

