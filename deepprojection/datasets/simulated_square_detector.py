#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import psana
import numpy as np
import random
import json
import csv
import h5py
import os
import inspect
import logging

from torch.utils.data import Dataset
from deepprojection.utils import set_seed, split_dataset

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
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class SPIImgDataset(Dataset):
    """
    SPI images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  
    """

    def __init__(self, config):
        self.fl_csv         = getattr(config, 'fl_csv'        , None)
        self.exclude_labels = getattr(config, 'exclude_labels', None)
        self.isflat         = getattr(config, 'isflat'        , None)
        self.dataset_usage  = getattr(config, 'dataset_usage' , None)    # train, validate, test, all
        self.frac_train     = getattr(config, 'frac_train'    , None)    # Proportion/Fraction of training examples
        self.frac_validate  = getattr(config, 'frac_validate' , None)    # Proportion/Fraction of validation examples
        self.seed           = getattr(config, 'seed'          , None)
        self.trans          = getattr(config, 'trans'         , None)

        self.h5_path_dict         = {}
        self.psana_imgreader_dict = {}
        self.imglabel_orig_list   = []

        # Constants
        self.KEY_TO_IMG = 'photons'

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Read the csv and collect files to read...
        self.h5_path_dict = {}
        hit_type_dict     = {}
        with open(self.fl_csv, 'r') as fh: 
            lines = csv.reader(fh)

            next(lines)

            for i, line in enumerate(lines):
                fl_base, label, drc_root = line
                basename = (fl_base, label)

                fl_h5 = f"{fl_base}.h5"
                path_h5 = os.path.join(drc_root, fl_h5)
                if not basename in self.h5_path_dict:
                    self.h5_path_dict[basename] = path_h5

                # Accumulate hit types and num of images associated...
                # [WORKAROUND] This works only if the same hit type implies the
                # same number of images.  
                hit_type = fl_base[fl_base.find('.') + 1 : ]
                if not hit_type in hit_type_dict: 
                    with h5py.File(path_h5, 'r') as fh:
                        num_imgs, _, _, _ = fh.get(self.KEY_TO_IMG).shape

                    hit_type_dict[hit_type] = num_imgs

        # Enumerate each image from all datasets, saved in h5 handles, w/o reading it...
        for basename, h5_path in self.h5_path_dict.items():
            fl_base, label = basename

            hit_type = fl_base[fl_base.find('.') + 1 : ]

            # Traverse every images in a synthetic dataset (it's okay to be slow)...
            for id_frame in range(hit_type_dict[hit_type]):
                self.imglabel_orig_list.append( (fl_base, id_frame, label) )

        ## # Split the original image list into training set and test set...
        ## num_list  = len(self.imglabel_orig_list)
        ## num_train = int(self.frac_train * num_list)

        ## # Get training examples
        ## imglabel_train_list = random.sample(self.imglabel_orig_list, num_train)

        ## # Get test examples
        ## imglabel_test_list = set(self.imglabel_orig_list) - set(imglabel_train_list)
        ## imglabel_test_list = sorted(list(imglabel_test_list))

        ## self.imglabel_list = imglabel_train_list if self.istrain else imglabel_test_list

        # Split original dataset into training and holdout...
        imglabel_train_list, imglabel_holdout_list = split_dataset(self.imglabel_orig_list, self.frac_train)

        # Calculate the percentage of validation in the whole holdout set...
        frac_holdout = 1.0 - self.frac_train
        frac_validate_in_holdout = self.frac_validate / frac_holdout if self.frac_validate is not None else 0.5

        # Split holdout dataset into validation and test...
        imglabel_valid_list, imglabel_test_list = split_dataset(imglabel_holdout_list, frac_validate_in_holdout)

        # Choose which dataset is going to be used, defaults to original set...
        dataset_by_usage_dict = {
            'train'    : imglabel_train_list,
            'validate' : imglabel_valid_list,
            'test'     : imglabel_test_list,
        }
        self.imglabel_list = self.imglabel_orig_list
        if self.dataset_usage in dataset_by_usage_dict:
            self.imglabel_list = dataset_by_usage_dict[self.dataset_usage]

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def get_img_and_label(self, idx):
        # Read image...
        fl_base, id_frame, label = self.imglabel_list[idx]
        basename = (fl_base, label)

        h5_path = self.h5_path_dict[basename]
        with h5py.File(h5_path, 'r') as fh:
            img = fh.get(self.KEY_TO_IMG)[id_frame][0]

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        return img, label


    def __getitem__(self, idx):
        img, label = self.get_img_and_label(idx)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img_norm = (img - img_mean) / img_std

        # If not flat, add one extra dimension to reflect the number channels...
        img_norm = img_norm[np.newaxis,] if not self.isflat else img_norm.reshape(-1)

        return img_norm, int(label)




class Siamese(SPIImgDataset):

    def __init__(self, config):
        super().__init__(config)

        self.size_sample = getattr(config, 'size_sample')

        # Create a lookup table for locating the sequence number (seqi) based on a label...
        label_seqi_dict = {}
        for seqi, (_, _, label) in enumerate(self.imglabel_list):
            # Keep track of label and its seqi
            if not label in label_seqi_dict: label_seqi_dict[label] = [seqi]
            else                           : label_seqi_dict[label].append(seqi)

        # Consolidate labels in the dataset...
        self.labels = sorted(list(set([ i[-1] for i in self.imglabel_list ])))

        # Log the number of images for each label...
        logger.info("___/ Dataset statistics \___")
        for label in self.labels:
            num_img = len(label_seqi_dict[label])
            logger.info(f"KV - {label:16s} : {num_img}")

        self.label_seqi_dict = { k: v for k, v in sorted(label_seqi_dict.items(), key=lambda item: item[0]) }

        return None




class SiameseDataset(Siamese):
    """
    Siamese requires an input of three images at a time, namely anchor,
    positive, and negative.  This dataset will create such triplet
    automatically by randomly choosing an anchor followed up by randomly
    selecting a positive and negative, respectively.
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        # Form triplet for ML training...
        self.triplets = self._form_triplets(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        id_anchor, id_pos, id_neg = self.triplets[idx]

        # Read the anchor, pos, neg...
        img_anchor, label_anchor = super().__getitem__(id_anchor)
        img_pos, _ = super().__getitem__(id_pos)
        img_neg, _ = super().__getitem__(id_neg)

        res = img_anchor, img_pos, img_neg, label_anchor

        # Append (fl_base, id_frame, label) to the result...
        for i in (id_anchor, id_pos, id_neg): 
            fl_base, id_frame, label = self.imglabel_list[i]
            title = f"{fl_base} {id_frame:>06d} {label}"
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
    Siamese testset returns a list of image pairs, anchor and second.  

    This implementation is mainly designed to measure true/false positve/negative.  

    Measurement: result_pred == result_supposed, result_pred
    Here result means 
    - match   : positve
    - mismatch: negative
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        # Form triplet for ML training...
        self.doublets = self._form_doublets(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        id_anchor, id_second = self.doublets[idx]

        # Read the anchor, pos, neg...
        img_anchor, label_anchor = super().__getitem__(id_anchor)
        img_second, _ = super().__getitem__(id_second)

        res = img_anchor, img_second, label_anchor

        # Append (fl_base, id_frame, label) to the result...
        for i in (id_anchor, id_second): 
            fl_base, id_frame, label = self.imglabel_list[i]
            title = f"{fl_base} {id_frame:>06d} {label}"
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




class MultiwayQueryset(Siamese):
    """
    Siamese testset for multi-way classification.

    This implementation is mainly designed to measure true/false positve/negative.  
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        # Form triplet for ML training...
        self.queryset = self._form_queryset(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        query_tuple = self.queryset[idx]
        id_query, ids_test = query_tuple[0], query_tuple[1:]

        # Read the query and test images...
        img_query, _ = super().__getitem__(id_query)
        imgs_test = []
        for id_test in ids_test:
            img_test, _ = super().__getitem__(id_test)
            imgs_test.append(img_test)

        res = [img_query, ] + imgs_test

        # Append (fl_base, id_frame, label) to the result...
        for i in query_tuple: 
            fl_base, id_frame, label = self.imglabel_list[i]
            title = f"{fl_base} {id_frame:>06d} {label}"
            res += (title, )

        return res


    def _form_queryset(self, label_seqi_dict):
        """ 
        Creating `size_sample` queryset that consists of one query image and 
        other random images selected from each label.

        Used for model validation only.  
        """
        # Select two list of random labels following uniform distribution...
        # For queryed image
        size_sample       = self.size_sample
        label_query_list = random.choices(self.labels, k = size_sample)

        # Form a queryset...
        queryset = []
        for label_query in label_query_list:
            # Fetch a bucket of query images...
            query_bucket = label_seqi_dict[label_query]

            # Randomly sample one query...
            id_query = random.choice(query_bucket)

            # Find a test image from each label...
            ids_test = []
            for label_test in self.labels:
                # Fetch a bucket of images for this label only...
                test_bucket = label_seqi_dict[label_test]

                # Randomly sample one test image...
                id_test = random.choice(test_bucket)
                ids_test.append(id_test)

            # Combine the query id with test ids...
            query_and_test = [id_query, ] + ids_test
            queryset.append( query_and_test )

        return queryset




class SimpleSet(Siamese):
    """
    Simple set feeds one example to model at a time.  The purpose is simply to
    encode each example for downstream analysis.  
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        # Form triplet for ML training...
        self.simpleset = self._form_simpleset(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        single_tuple = self.simpleset[idx]
        id_single    = single_tuple

        # Read the single image...
        img_single, _ = super().__getitem__(id_single)
        res = (img_single, )

        # Append (fl_base, id_frame, label) to the result...
        fl_base, id_frame, label = self.imglabel_list[id_single]
        title = f"{fl_base} {id_frame:>06d} {label}"
        res += (title, )

        return res


    def _form_simpleset(self, label_seqi_dict):
        """ 
        Creating `size_sample` simple set that consists of one image only. 
        """
        # Select two list of random labels following uniform distribution...
        # For a single image
        size_sample = self.size_sample
        label_list  = random.choices(self.labels, k = size_sample)

        # Form a simple set...
        simpleset = []
        for label in label_list:
            # Fetch a bucket of images...
            bucket = label_seqi_dict[label]

            # Randomly sample one...
            id = random.choice(bucket)

            simpleset.append(id)

        return simpleset




class OnlineDataset(Siamese):
    """
    For online leraning.  
    """

    def __init__(self, config):
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        # Form triplet for ML training...
        self.online_set = self._form_online_set(label_seqi_dict)

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        single_tuple = self.online_set[idx]
        id_single    = single_tuple

        # Read the single image...
        img_single, label_single = super().__getitem__(id_single)
        res = (img_single, label_single)

        # Append (fl_base, id_frame, label) to the result...
        fl_base, id_frame, label = self.imglabel_list[id_single]
        title = f"{fl_base} {id_frame:>06d} {label}"
        res += (title, )

        return res


    def _form_online_set(self, label_seqi_dict):
        """ 
        Creating `size_sample` simple set that consists of one image only. 
        """
        # Select two list of random labels following uniform distribution...
        # For a single image
        size_sample = self.size_sample
        label_list  = random.choices(self.labels, k = size_sample)

        # Form a simple set...
        online_set = []
        for label in label_list:
            # Fetch a bucket of images...
            bucket = label_seqi_dict[label]

            # Randomly sample one...
            id = random.choice(bucket)

            online_set.append(id)

        return online_set


    def report(self, verbose = False):
        # Log the number of images for each label...
        logger.info("___/ List of entries in dataset \___")

        count_per_label_dict = {}
        for idx in self.online_set:
            fl_base, id_frame, label = self.imglabel_list[idx]
            if verbose: 
                logger.info(f"ENTRIES - {fl_base:16s} {int(id_frame):06d} {label:2s}")

            if not label in count_per_label_dict: count_per_label_dict[label]  = 1
            else                                : count_per_label_dict[label] += 1

        for label, count in count_per_label_dict.items():
            logger.info(f"COUNTS - label {label} : {count}")
