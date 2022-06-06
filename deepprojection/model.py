#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
from deepprojection.utils import calc_dmat
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
        self.alpha   = config.alpha
        self.encoder = config.encoder


    def forward(self, img_anchor, img_pos, img_neg):
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

        return img_anchor_embed, img_pos_embed, img_neg_embed, loss_triplet.mean()


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr = config_train.lr)
        ## optimizer = torch.optim.SGD(self.encoder.parameters(), lr = config_train.lr)

        return optimizer




class OnlineSiameseModel(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.alpha   = config.alpha
        self.encoder = config.encoder


    def forward(self, batch_imgs, batch_labels, batch_titles, 
                is_logging = True, 
                method = 'semi-hard',
                shuffle = False):
        # Supposed methods
        select_method_dict = {
            ## 'batch-hard'       : self.batch_hard,
            'semi-hard'        : self.batch_semi_hard,
            'random-semi-hard' : self.batch_random_semi_hard,
            'random'           : self.batch_random,
        }

        # Assert a valid method...
        assert method in select_method_dict, f"{method} is not supported!!!"

        # Select triplets...
        select_method = select_method_dict[method]
        triplets = select_method(batch_imgs, batch_labels, batch_titles, is_logging = is_logging, shuffle = shuffle)

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


    def batch_random(self, batch_imgs, batch_labels, batch_titles, is_logging = True, shuffle = False, **kwargs):
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

        if shuffle: 
            idx_shuffle_list = random.sample(range(len(triplets)), k = len(triplets))
            triplets = [ triplets[i] for i in idx_shuffle_list ]

        if is_logging:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                title_achr = batch_titles[batch_idx_achr]
                title_pos  = batch_titles[batch_idx_pos]
                title_neg  = batch_titles[batch_idx_neg]
                logger.info(f"DATA - {title_achr} {title_pos} {title_neg}")

        return triplets


    def batch_random_semi_hard(self, batch_imgs, batch_labels, batch_titles, is_logging = True, shuffle = False, **kwargs):
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

        if shuffle: 
            idx_shuffle_list = random.sample(range(len(triplets)), k = len(triplets))
            triplets = [ triplets[i] for i in idx_shuffle_list ]
            dist_log = [ dist_log[i] for i in idx_shuffle_list ]

        if is_logging:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                title_achr = batch_titles[batch_idx_achr]
                title_pos  = batch_titles[batch_idx_pos]
                title_neg  = batch_titles[batch_idx_neg]
                dist_pos   = dist_log[idx][0]
                dist_neg   = dist_log[idx][1]
                logger.info(f"DATA - {title_achr} {title_pos} {title_neg} {dist_pos:12.6f} {dist_neg:12.6f}")

        return triplets



    def batch_semi_hard(self, batch_imgs, batch_labels, batch_titles, is_logging = True, shuffle = False, **kwargs):
        ''' The idea is to go through each one in batch_imgs and find all
            positive images, and then following it up with selecting a negative
            case that still satisfyies dn > dp (semi hard cases).  
        '''
        # Encode all batched images without autograd tracking...
        with torch.set_grad_enabled(False):
            batch_embs = self.encoder.encode(batch_imgs)

        # Convert batch labels to dictionary for fast lookup...
        batch_label_dict = {}
        batch_label_list = batch_labels.cpu().numpy()
        for i, v in enumerate(batch_label_list):
            if not v in batch_label_dict: batch_label_dict[v] = [i]
            else                        : batch_label_dict[v].append(i)

        # Figure out total number of positive pairs...
        num_pos_track = 0
        for k, v in batch_label_dict.items():
            num_pos_track += len(list(permutations(v, 2)))

        # ___/ NEGATIVE MINIG \___
        # Go through each image in the batch and form triplets...
        # Prepre for logging
        triplets = []
        dist_log = []
        for batch_idx_achr, img in enumerate(batch_imgs):
            # Get the label of the image...
            batch_label_achr = batch_label_list[batch_idx_achr]

            # Find all instances that have the same labels...
            batch_idx_pos_list = batch_label_dict[batch_label_achr]

            # Go through each positive image and find the semi hard negative...
            for batch_idx_pos in batch_idx_pos_list:
                # Ignore trivial cases...
                if batch_idx_achr == batch_idx_pos: continue

                # Find positive embedding squared distances..
                emb_achr = batch_embs[batch_idx_achr]
                emb_pos  = batch_embs[batch_idx_pos]
                delta_emb_pos = emb_achr - emb_pos
                dist_pos = torch.sum(delta_emb_pos * delta_emb_pos)

                # Retrieve all negative candidates...
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

        if shuffle: 
            idx_shuffle_list = random.sample(range(len(triplets)), k = len(triplets))
            triplets = [ triplets[i] for i in idx_shuffle_list ]
            dist_log = [ dist_log[i] for i in idx_shuffle_list ]

        if is_logging:
            # Logging all cases...
            for idx, triplet in enumerate(triplets):
                batch_idx_achr, batch_idx_pos, batch_idx_neg = triplet
                title_achr = batch_titles[batch_idx_achr]
                title_pos  = batch_titles[batch_idx_pos]
                title_neg  = batch_titles[batch_idx_neg]
                dist_pos   = dist_log[idx][0]
                dist_neg   = dist_log[idx][1]
                logger.info(f"DATA - {title_achr} {title_pos} {title_neg} {dist_pos:12.6f} {dist_neg:12.6f}")

        return triplets


    def batch_hard(self, batch_imgs, batch_labels, batch_titles, **kwargs):
        # Get batch size...
        batch_size = len(batch_labels)

        # Encode all batched images...
        batch_embs = self.encoder.encode(batch_imgs)

        # Get distance matrix...
        dmat = calc_dmat(batch_embs, batch_embs, is_sqrt = False)

        # Get the bool matrix...
        bmat = batch_labels[:, None] == batch_labels[:, None].t()

        # ___/ MINE HARDEST POSITIVES \___
        # Get the dmat masked by bmat...
        dmat_masked_positive = dmat * bmat

        # Get the row-wise max distances and their indices...
        batch_hardest_positives = torch.max(dmat_masked_positive, dim = -1)

        # ___/ MINE HARDEST NEGATIVES \___
        # Assign positive distances the max distance to facilitate mining min values...
        MAX_DIST = torch.max(dmat)
        dmat_masked_negative = dmat * (~bmat)
        dmat_masked_negative[bmat] = MAX_DIST

        # Get the row-wise min distances and their indices...
        batch_hardest_negatives = torch.min(dmat_masked_negative, dim = -1)

        ## # Log hardest examples...
        ## batch_idx_hardest_positive = batch_hardest_positives.indices
        ## batch_idx_hardest_negative = batch_hardest_negatives.indices
        ## batch_val_hardest_positive = batch_hardest_positives.values
        ## batch_val_hardest_negative = batch_hardest_negatives.values

        ## for i in range(len(batch_labels)):
        ##     # Retrieve the title of hardest example...
        ##     title_hardest_positive = batch_titles[ batch_idx_hardest_positive[i] ]
        ##     title_hardest_negative = batch_titles[ batch_idx_hardest_negative[i] ]
        ##     val_hardest_positive = batch_val_hardest_positive[i]
        ##     val_hardest_negative = batch_val_hardest_negative[i]
        ##     logger.info(f"DATA - {batch_titles[i]} {title_hardest_positive} {title_hardest_negative} {val_hardest_positive:12.6f} {val_hardest_negative:12.6f}")

        # Get the batch hard row-wise triplet loss...
        batch_loss_triplet = torch.relu(batch_hardest_positives.values - batch_hardest_negatives.values + self.alpha)

        return batch_loss_triplet.mean()


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr = config_train.lr)
        ## optimizer = torch.optim.SGD(self.encoder.parameters(), lr = config_train.lr)

        return optimizer




class SiameseModelCompare(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.encoder = config.encoder


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
