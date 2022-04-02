#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
from deepprojection.utils import calc_dmat
from itertools import combinations

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


    def forward(self, batch_imgs, batch_labels, batch_titles):
        ''' The idea is to go through each one in batch_imgs and find all
            positive images, and then following it up with selecting a negative
            case that still satisfyies dn > dp (semi hard cases).  
        '''
        # Encode all batched images...
        batch_embs = self.encoder.encode(batch_imgs)

        # Convert batch labels to dictionary for fast lookup...
        label_dict = {}
        batch_label_list = batch_labels.cpu().numpy()
        for i, v in enumerate(batch_label_list):
            if not v in label_dict: label_dict[v] = [i]
            else                  : label_dict[v].append(i)

        # Figure out total number of positive pairs...
        num_pos_track = 0
        for k, v in label_dict.items():
            num_pos_track += len(list(combinations(v, 2)))

        # ___/ NEGATIVE MINIG \___
        # Go through each image in the batch and form triplets...
        triplets = []
        dist_raw = []
        dist_pair_list = torch.zeros(num_pos_track)
        dist_idx_track = 0
        for idx_achr, img in enumerate(batch_imgs):
            # Get the label of the image...
            label_achr = batch_label_list[idx_achr]

            # Find all instances that have the same labels...
            idx_pos_list = label_dict[label_achr]

            # Go through each positive image and find the semi hard negative...
            for idx_pos in idx_pos_list:
                # Ignore trivial cases...
                if idx_achr >= idx_pos: continue

                # Find positive embedding squared distances..
                emb_achr = batch_embs[idx_achr]
                emb_pos  = batch_embs[idx_pos]
                delta_emb_pos = emb_achr - emb_pos
                dist_pos = torch.sum(delta_emb_pos * delta_emb_pos)

                # Retrieve all negative candidates...
                idx_neg_list = []
                for label, idx_list in label_dict.items():
                    if label == label_achr: continue
                    idx_neg_list += idx_list

                # Collect all negative embeddings...
                emb_neg_list = batch_embs[idx_neg_list]

                # Find negative embedding squared distances...
                delta_emb_neg_list = emb_achr[None, :] - emb_neg_list
                dist_neg_list = torch.sum( delta_emb_neg_list * delta_emb_neg_list, dim = -1 )

                # Find negative squared distance satisfying dist_neg > dist_pos (semi hard)...
                cond_semihard = dist_neg_list > dist_pos

                # If semi hard exists???
                if torch.any(cond_semihard):
                    # Look for the argmin...
                    min_neg_semihard = torch.min(dist_neg_list[cond_semihard], dim = -1)
                    idx_neg  = idx_neg_list[ min_neg_semihard.indices ]
                    dist_neg = min_neg_semihard.values

                # Otherwise, randomly select one negative example???
                else:
                    idx_reduced = random.choice(range(len(idx_neg_list)))
                    idx_neg  = idx_neg_list [idx_reduced]
                    dist_neg = dist_neg_list[idx_reduced]

                # Track triplet...
                triplets.append((idx_achr, idx_pos, idx_neg))
                dist_raw.append((dist_pos, dist_neg))

                # Track triplet loss...
                ## dist_pair_list[idx_achr] = torch.tensor([dist_pos, -dist_neg])
                dist_pair_list[dist_idx_track] = dist_pos - dist_neg
                dist_idx_track += 1

        # Logging all cases...
        for triplet in triplets:
            idx_achr, idx_pos, idx_neg = triplet
            title_achr = batch_titles[idx_achr]
            title_pos  = batch_titles[idx_pos]
            title_neg  = batch_titles[idx_neg]
            dist_pos   = dist_raw[idx_achr][0]
            dist_neg   = dist_raw[idx_achr][1]
            logger.info(f"DATA - {title_achr} {title_pos} {title_neg} {dist_pos:12.6f} {dist_neg:12.6f}")

        # Compute trilet loss, apply relu and return the mean triplet loss...
        ## triplet_loss = torch.sum(dist_pair_list, dim = -1) + self.alpha
        triplet_loss = dist_pair_list + self.alpha
        return torch.relu(triplet_loss).mean()


    def batch_hard_forward(self, batch_imgs, batch_labels, batch_titles):
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


    def forward(self, img_anchor, img_second):
        # Encode images...
        img_anchor_embed = self.encoder.encode(img_anchor)
        img_second_embed = self.encoder.encode(img_second)

        # Calculate the RMSD between anchor and second...
        img_diff           = img_anchor_embed - img_second_embed
        rmsd_anchor_second = torch.sum(img_diff * img_diff, dim = -1)

        return img_anchor_embed, img_second_embed, rmsd_anchor_second.mean()




class SimpleEmbedding(nn.Module):
    """ Embedding independent triplet loss. """

    def __init__(self, config):
        super().__init__()
        self.encoder = config.encoder


    def forward(self, img):
        return self.encoder.encode(img)
