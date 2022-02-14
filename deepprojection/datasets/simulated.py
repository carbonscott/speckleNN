#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load PyTorch
import torch
from torch.utils.data import Dataset

# Load misc modules
import numpy as np
import random
import os
import logging

from deepprojection.utils import downsample

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''
    NOHIT      = '0'
    SINGLE     = '1'
    MULTI      = '2'
    UNKNOWN    = '3'
    NEEDHELP   = '4'
    BACKGROUND = '9'

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"{k:16s} : {v}")


class Siamese:

    def __init__(self, config):
        self.x_train = np.load(config.path_x_train)
        self.y_train = np.load(config.path_y_train)

        self.num_stockimgs = len(self.y_train)
        self.size_sample   = config.size_sample

        # Consolidate labels in the dataset...
        self.labels = list(set(self.y_train))

        # Create a lookup table for locating the sequence number (seqi) based on a label
        label_seqi_dict = {}
        for seqi, label in enumerate(self.y_train):
            if not label in label_seqi_dict: label_seqi_dict[label] = [seqi]
            else: label_seqi_dict[label].append(seqi)

        self.label_seqi_dict = label_seqi_dict

        return None




class SiameseDataset(Siamese):
    """
    Siamese requires an input of three images at a time, namely anchor,
    positive, and negative.  This dataset will create such triplet
    automatically by randomly choosing an anchor followed up by randomly
    selecting a positive and negative, respectively.
    """

    def __init__(self, config):
        super.__init__(config)

        label_seqi_dict = self.label_seqi_dict

        self.triplets = self._form_triplets(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        id_anchor, id_pos, id_neg = self.triplets[idx]

        # Read the anchor, pos, neg
        img_anchor   = self.x_train[id_anchor]
        img_pos      = self.x_train[id_pos]
        img_neg      = self.x_train[id_neg]
        label_anchor = self.y_train[id_anchor]

        res = img_anchor, img_pos, img_neg, label_anchor

        # Append (exp, run, event_num, label) to the result
        for i in (id_anchor, id_pos, id_neg): 
            title = f"{i} {self.y_train[i]}"
            res += (title, )

        return res


    def _form_triplets(self, label_seqi_dict):
        """ 
        Creating `size_sample` tripets of id_anchor, id_pos, id_neg.  

        The anchor label is sampled from a set of labels, which ensures that
        the triplets don't have a favored combination.  
        """
        size_sample       = self.size_sample
        label_anchor_list = random.choices(self.labels, k = size_sample)

        # Collection of triplets...
        triplets = []
        for label_anchor in label_anchor_list:
            # Fetch the anchor label...
            # Create buckets of anchors...
            anchor_bucket = label_seqi_dict[label_anchor]

            # Randomly sample one anchor...
            id_anchor = random.choice(anchor_bucket)

            # Create buckets of positives according to the anchor...
            pos_bucket = anchor_bucket

            # Randomly sample one positive...
            id_pos = random.choice(pos_bucket)

            # Create buckets of negatives according to the anchor...
            neg_labels = [ label for label in self.labels if label != label_anchor ]
            neg_label  = random.choice(neg_labels)
            neg_bucket = label_seqi_dict[neg_label]

            # Randomly sample one negative...
            id_neg = random.choice(neg_bucket)

            triplets.append( (id_anchor, id_pos, id_neg) )

        return triplets




class SiameseTestset(Siamese):
    """
    Siamese requires an input of three images at a time, namely anchor,
    positive, and negative.  This dataset will create such triplet
    automatically by randomly choosing an anchor followed up by randomly
    selecting a positive and negative, respectively.
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        self.doublets = self._form_doublets(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        id_anchor, id_second = self.doublets[idx]

        # Read the anchor, pos, neg
        img_anchor   = self.x_train[id_anchor]
        img_second   = self.x_train[id_second]
        label_anchor = self.y_train[id_anchor]

        res = img_anchor, img_second, label_anchor

        # Append (exp, run, event_num, label) to the result
        for i in (id_anchor, id_second): 
            title = f"{i} {self.y_train[i]}"
            res += (title, )

        return res


    def _form_doublets(self, label_seqi_dict):
        """ 
        Creating `size_sample` doublets of two images.

        Used for model validation only.  
        """
        # Select two list of random labels following uniform distribution...
        # For anchor
        size_sample       = self.size_sample
        label_anchor_list = random.choices(self.labels, k = size_sample)

        # Collection of doublets...
        doublets = []
        for label_anchor in label_anchor_list:
            # Fetch the anchor label...
            # Create buckets of anchors...
            anchor_bucket = label_seqi_dict[label_anchor]

            # Randomly sample one anchor...
            id_anchor = random.choice(anchor_bucket)

            # Create buckets of second images...
            label_second = random.choice(self.labels)

            second_bucket = label_seqi_dict[label_second]

            # Randomly sample one second image...
            id_second = random.choice(second_bucket)

            doublets.append( (id_anchor, id_second) )

        return doublets
