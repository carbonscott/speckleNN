#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import os
import logging

from torch.utils.data import Dataset

from ..utils import set_seed

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




class SPIDataset(Dataset):
    """
    get_     method returns data by sequential index.
    extract_ method returns object by files.
    """

    def __init__(self, dataset_list, trans = None):
        self.dataset_list = dataset_list
        self.trans        = trans

        return None


    def __len__(self):
        return len(self.dataset_list)


    def __getitem__(self, idx):
        img, label, title = self.dataset_list[idx]

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            img = self.trans(img)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img      = (img - img_mean) / img_std

        return img[None,], label, title




class SPIOnlineDataset(Dataset):
    """
    For online learning.
    dataset_list element:
        (img, label, title_tuple)
    """

    def __init__(self, dataset_list, 
                       size_sample, 
                       size_sample_per_class = None, 
                       trans                 = None, 
                       allows_cache_trans    = False,
                       seed                  = None):
        # Unpack parameters...
        self.size_sample           = size_sample
        self.size_sample_per_class = size_sample_per_class
        self.dataset_list          = dataset_list
        self.trans                 = trans
        self.allows_cache_trans    = allows_cache_trans
        self.seed                  = seed

        # Set seed for data spliting...
        if seed is not None:
            set_seed(seed)

        self.random_state_cache_dict = {}

        self.dataset_cache_dict = {}

        # Fetch all metadata...
        self.metadata_list = [ metadata for _, _, metadata in self.dataset_list ]

        # Create a lookup table for locating the sequence number (seqi) based on a label...
        label_seqi_dict = {}
        for seqi, (_, label, _) in enumerate(self.dataset_list):
            # Keep track of label and its seqi
            if not label in label_seqi_dict: label_seqi_dict[label] = [seqi]
            else                           : label_seqi_dict[label].append(seqi)
        self.label_seqi_dict = label_seqi_dict

        # Get unique labels...
        self.labels = sorted(list(set([ label for _, label, _ in self.dataset_list ])))

        # Form triplet for ML training...
        self.online_set = self._form_online_set()

        return None


    def __len__(self):
        return self.size_sample


    def get_random_state(self):
        state_random = (random.getstate(), np.random.get_state())

        return state_random


    def set_random_state(self, state_random):
        state_random, state_numpy = state_random
        random.setstate(state_random)
        np.random.set_state(state_numpy)

        return None


    def cache_dataset(self):
        for idx in range(self.size_sample):
            if idx in self.dataset_cache_dict: continue

            img, label, metadata = self.get_data(idx)
            self.dataset_cache_dict[idx] = (img, label, metadata)

        return None


    def get_data(self, idx):
        # Retrive a sampled image...
        idx_sample = self.online_set[idx]
        img, label, metadata = self.dataset_list[idx_sample]

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            if self.allows_cache_trans:
                # Memorize the random state by index
                if idx not in self.random_state_cache_dict:
                    state_random = self.get_random_state()
                    self.random_state_cache_dict[idx] = state_random
                state_random = self.random_state_cache_dict[idx]
                self.set_random_state(state_random)

            img = self.trans(img)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img      = (img - img_mean) / img_std

        return img, label, metadata


    def __getitem__(self, idx):
        idx_sample = self.online_set[idx]
        img, label, metadata = self.dataset_cache_dict[idx] if idx in self.dataset_cache_dict \
                                                            else self.get_data(idx)

        return img[None,], label, metadata


    def _form_online_set(self):
        """ 
        Creating `size_sample` simple set that consists of one image only. 
        """
        # Select two list of random labels following uniform distribution...
        # For a single image
        size_sample = self.size_sample
        label_list  = random.choices(self.labels, k = size_sample)

        # Limit unique samples per class...
        label_seqi_dict = self.label_seqi_dict
        label_seqi_sampled_dict = {}
        if self.size_sample_per_class is not None:
            for label in self.labels:
                # Fetch a bucket of images...
                bucket = label_seqi_dict[label]

                # Randomly sample certain number of unique examples per class...
                num_sample = min(self.size_sample_per_class, len(bucket))
                id_list = random.sample(bucket, num_sample)

                label_seqi_sampled_dict[label] = id_list

            label_seqi_dict = label_seqi_sampled_dict

        self.label_seqi_dict = label_seqi_dict

        # Form a simple set...
        online_set = []
        for label in label_list:
            # Fetch a bucket of images...
            bucket = label_seqi_dict[label]

            # Randomly sample one...
            id = random.choice(bucket)

            online_set.append(id)

        return online_set


    def report(self):
        # Log the number of images for each label...
        logger.info("___/ List of entries in dataset \___")

        event_label_dict = {}
        for idx in self.online_set:
            _, _, _, label = self.metadata_list[idx]

            if not label in event_label_dict: event_label_dict[label] = [ idx ]
            else                            : event_label_dict[label].append(idx)

        for label, idx_list in event_label_dict.items():
            count = len(idx_list)
            logger.info(f"KV - (event count) label {label} : {count}")

        for label, idx_list in event_label_dict.items():
            count = len(set(idx_list))
            logger.info(f"KV - (unique event count) label {label} : {count}")




class MultiwayQueryset(Dataset):
    """
    Siamese testset for multi-way classification.

    This implementation is mainly designed to measure true/false positve/negative.  
    """

    def __init__(self, dataset_list, 
                       size_sample, 
                       size_sample_per_class = None, 
                       trans                 = None, 
                       allows_cache_trans    = False,
                       seed                  = None):
        # Unpack parameters...
        self.size_sample           = size_sample
        self.size_sample_per_class = size_sample_per_class
        self.dataset_list          = dataset_list
        self.trans                 = trans
        self.allows_cache_trans    = allows_cache_trans
        self.seed                  = seed

        # Set seed for data spliting...
        if seed is not None:
            set_seed(seed)

        self.random_state_cache_dict = {}

        # Fetch all metadata...
        self.metadata_list = [ metadata for _, _, metadata in self.dataset_list ]

        # Create a lookup table for locating the sequence number (seqi) based on a label...
        label_seqi_dict = {}
        for seqi, (_, label, _) in enumerate(self.dataset_list):
            # Keep track of label and its seqi
            if not label in label_seqi_dict: label_seqi_dict[label] = [seqi]
            else                           : label_seqi_dict[label].append(seqi)
        self.label_seqi_dict = label_seqi_dict

        # Get unique labels...
        self.labels = sorted(list(set([ label for _, label, _ in self.dataset_list ])))


        # Form triplet for ML training...
        self.queryset = self._form_queryset()

        return None


    def __len__(self):
        return self.size_sample


    def __getitem__(self, idx):
        query_tuple = self.queryset[idx]
        id_query, ids_test = query_tuple[0], query_tuple[1:]

        # Read the query and test images...
        img_query, _, _ = self.dataset_list[id_query]
        imgs_test = []
        for id_test in ids_test:
            img_test, _, _ = self.dataset_list[id_test]
            imgs_test.append(img_test)

        # Apply any possible transformation...
        # How to define a custom transform function?
        # Input : img, **kwargs 
        # Output: img_transfromed
        if self.trans is not None:
            if self.allows_cache_trans:
                # Memorize the random state by index
                if idx not in self.random_state_cache_dict:
                    state_random = self.get_random_state()
                    self.random_state_cache_dict[idx] = state_random
                state_random = self.random_state_cache_dict[idx]
                self.set_random_state(state_random)

            img_query = self.trans(img_query)
            imgs_test = [ self.trans(img) for img in imgs_test ]

        img_query = self.normalize(img_query)
        imgs_test = [ self.normalize(img)[None,] for img in imgs_test ]

        res = [img_query[None,], ] + imgs_test

        # Append (exp, run, event_num, label) to the result
        for i in query_tuple: 
            title = self.metadata_list[i]
            res += [' '.join(title), ]

        return res


    def normalize(self, img):
        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img      = (img - img_mean) / img_std

        return img


    def _form_queryset(self):
        """ 
        Creating `size_sample` queryset that consists of one query image and 
        other random images selected from each label.

        Used for model validation only.  
        """
        # Select two list of random labels following uniform distribution...
        # For queryed image
        size_sample      = self.size_sample
        label_query_list = random.choices(self.labels, k = size_sample)

        # Form a queryset...
        queryset = []
        for label_query in label_query_list:
            # Fetch a bucket of query images...
            query_bucket = self.label_seqi_dict[label_query]

            # Randomly sample one query...
            id_query = random.choice(query_bucket)

            # Find a test image from each label...
            ids_test = []
            for label_test in self.labels:
                # Fetch a bucket of images for this label only...
                test_bucket = self.label_seqi_dict[label_test]

                # Randomly sample one test image...
                id_test = random.choice(test_bucket)
                ids_test.append(id_test)

            # Combine the query id with test ids...
            query_and_test = [id_query, ] + ids_test
            queryset.append( query_and_test )

        return queryset


    def get_random_state(self):
        state_random = (random.getstate(), np.random.get_state())

        return state_random


    def set_random_state(self, state_random):
        state_random, state_numpy = state_random
        random.setstate(state_random)
        np.random.set_state(state_numpy)

        return None


