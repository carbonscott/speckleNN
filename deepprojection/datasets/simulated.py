#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load PyTorch
import torch
from torch.utils.data import Dataset

# Load misc modules
import numpy as np
import random
import os


class SiameseDataset:
    """
    Siamese requires an input of three images at a time, namely anchor,
    positive, and negative.  This dataset will create such triplet
    automatically by randomly choosing an anchor followed up by randomly
    selecting a positive and negative, respectively.
    """

    def __init__(self, size_sample, debug = False):
        drc_data   = os.path.dirname(os.path.realpath(__file__))
        fl_x_train = os.path.join(drc_data, "x_train.npy")
        fl_y_train = os.path.join(drc_data, "y_train.npy")
        self.x_train    = np.load(fl_x_train)
        self.y_train    = np.load(fl_y_train)

        self.num_stockimgs = len(self.y_train)
        self.size_sample   = size_sample
        self.debug         = debug

        # Create a lookup table for locating the sequence number (seqi) based on a label
        self.label_seqi_dict = {}
        for seqi, label in enumerate(self.y_train):
            if not label in self.label_seqi_dict: self.label_seqi_dict[label] = [seqi]
            else: self.label_seqi_dict[label].append(seqi)

        self.triplets = self._form_tripets()

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

        if self.debug: 
            # Append (exp, run, event_num, label) to the result
            for i in (id_anchor, id_pos, id_neg): 
                title = f"{i} {self.y_train[i]}"
                res += (title, )

        return res


    def _form_tripets(self):
        """ Creating `size_sample` tripets of id_anchor, id_pos, id_neg"""
        # Randomly select an anchor `size_sample` times
        size_sample   = self.size_sample
        anchor_bucket = range(self.num_stockimgs)
        ids_anchor    = random.choices(anchor_bucket, k = size_sample)

        # Collection of triplets
        triplets = []
        for id_anchor in ids_anchor:
            # Fetch the anchor label
            label_anchor = self.y_train[id_anchor]

            # Create buckets of positives according to the anchor
            pos_bucket = self.label_seqi_dict[label_anchor]

            # Create buckets of negatives according to the anchor
            neg_bucket = []
            for label, ids in self.label_seqi_dict.items(): 
                if label == label_anchor: continue
                neg_bucket += ids

            # Randomly sample one positive and one negative
            id_pos = random.sample(pos_bucket, 1)[0]
            id_neg = random.sample(neg_bucket, 1)[0]

            triplets.append( (id_anchor, id_pos, id_neg) )

        return triplets
