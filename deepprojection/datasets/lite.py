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
        img, label, metadata = self.dataset_list[idx]

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

        return img[None,], label, metadata




class CustomData:

    def __init__(self, label, idx_list):
        self.label    = label
        self.idx_list = idx_list


    def __repr__(self):
        msg  = "label => "
        msg += f"({self.label[0]:d}, {self.label[1]})"
        msg += "\n"
        msg += "idx_list => "
        msg += " ".join([ f"{i:d}" for i in self.idx_list ])
        msg += "\n"

        return msg



class TripletCandidate(Dataset):
    '''
    """
    TripletCandidate(dataset_list, num_sample, num_sample_per_label, trans)

    This class will reutrn a list of `(encode, candidate_list)` so that PyTorch
    DataLoader can pack `label` and `candidate_list` into a batch,
    respectively.  This huge batch can then be divided into a number of
    mini-batch for model training and validation.

    Attributes
    ----------
    label_to_idx_dict : dict
        Dictionary of `label` to `index` mapping, in which `index` is used to
        access data in the `dataset_list`.

    label_to_encode_dict : dict
        Dictionary of `label` to `encode` mapping, where `encode` is an integer
        that encodes a label like `(1, '6Q5U')`.

    encode_to_label_dict : dict
        Dictionary with key-value pair in `label_to_encode_dict` but reversed.

    label_list : list
        List of all values of `label`.

    encode_list : list
        List of all values of `encode`.

    sample_list : list List of all sample that can be returned by the __call__
    function with a supplied index.  Each sample is a tuple of two elements --
    `(encode, candidate_list)`.

    Parameters
    ----------

    dataset_list : list
        List of data points, where a data point is defined as a tuple of three
        elements -- `(image, label, metadata)`.

    num_sample : int
        Number of samples.  See the comment section of `sample_list` for the
        definition of sample.

    num_sample_per_label : int
        Number of samples to be associated with one label.

    trans : list
        List of functions that transform an image.

    Examples
    --------
    >>>

    """
    '''

    def __init__(self, dataset_list,
                       num_sample            = 2,
                       num_sample_per_label  = 2,
                       trans                 = None,
                       mpi_comm              = None):
        self.dataset_list          = dataset_list
        self.num_sample            = num_sample
        self.num_sample_per_label = num_sample_per_label
        self.trans                 = trans
        self.mpi_comm              = mpi_comm

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

        self.label_to_idx_dict = self.build_label_to_idx_dict()

        self.label_to_encode_dict, self.encode_to_label_dict = self.encode_label()

        self.label_list  = []
        self.encode_list = []
        for label, encode in self.label_to_idx_dict.items():
            self.label_list.append(label)
            self.encode_list.append(encode)

        self.sample_list = self.build_sample_list()

        self.dataset_cache_dict = {}


    def encode_label(self):
        label_to_encode_dict = { label  : encode for encode, label in enumerate(self.label_to_idx_dict.keys()) }
        encode_to_label_dict = { encode : label  for encode, label in enumerate(self.label_to_idx_dict.keys()) }

        return label_to_encode_dict, encode_to_label_dict


    def build_label_to_idx_dict(self):
        label_to_idx_dict = {}
        for idx, (img, label, metadata) in enumerate(self.dataset_list):
            if label not in label_to_idx_dict: label_to_idx_dict[label] = []
            label_to_idx_dict[label].append(idx)

        return label_to_idx_dict


    def build_sample_list(self):
        sample_list = []
        for idx in range(self.num_sample):
            label  = random.choice(self.label_list)
            encode = self.label_to_encode_dict[label]
            candidate_list = random.choices(self.label_to_idx_dict[label],
                                               k = self.num_sample_per_label)
            sample_list.append( (encode, candidate_list) )

        return sample_list


    def __len__(self):
        return self.num_sample


    def get_sample(self, idx):
        encode, candidate_list = self.sample_list[idx]

        ## return torch.tensor(encode), torch.tensor(candidate_list)
        return encode, candidate_list


    def get_data(self, idx):
        encode, candidate_list = self.get_sample(idx)

        # Loop through all candidate by index and create a candidate tensor...
        metadata_list = []
        num_candidate = len(candidate_list)
        for i, candidate in enumerate(candidate_list):
            # Fetch original data...
            img, label, metadata = self.dataset_list[candidate]

            # Apply any possible transformation...
            # How to define a custom transform function?
            # Input : img, **kwargs 
            # Output: img_transfromed
            if self.trans is not None:
                img = self.trans(img)

            if i == 0:
                # Preallocate a matrix to hold all data...
                size_y, size_x = img.shape[-2:]
                img_nplist = np.zeros((num_candidate, 1, size_y, size_x), dtype = np.float32)
                #                                     ^
                # Torch Channel ______________________|

            # Keep img in memory...
            img_nplist[i] = img

            # Save metadata...
            metadata_list.append(metadata)

        return encode, img_nplist, metadata_list


    def __getitem__(self, idx):
        encode, img_nplist, metadata_list =                                \
            self.dataset_cache_dict[idx] if idx in self.dataset_cache_dict \
                                         else                              \
                                         self.get_data(idx)

        return encode, img_nplist, metadata_list


    def mpi_cache_dataset(self):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from ..utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        idx_list = range(self.num_sample)

        # Split the workload...
        idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

        # Process chunk by each worker...
        # No need to sync the dataset_cache_dict across workers
        if mpi_rank != 0:
            if mpi_rank < len(idx_list_in_chunk):
                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                self.dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

            mpi_comm.send(self.dataset_cache_dict, dest = 0, tag = mpi_data_tag)

        if mpi_rank == 0:
            idx_list_per_worker = idx_list_in_chunk[mpi_rank]
            self.dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

            for i in range(1, mpi_size, 1):
                dataset_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)

                self.dataset_cache_dict.update(dataset_cache_dict)

        return None


    def _mpi_cache_data_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        dataset_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in dataset_cache_dict: continue

            print(f"Cacheing data point {idx}...")

            encode, img_nplist, metadata_list = self.get_data(idx)
            dataset_cache_dict[idx] = (encode, img_nplist, metadata_list)

        return dataset_cache_dict




class SPIOnlineDataset(Dataset):
    """
    For online learning.
    dataset_list element:
        (img, label, metadata_tuple)
    """

    def __init__(self, dataset_list, 
                       size_sample, 
                       size_sample_per_class = None, 
                       trans                 = None, 
                       allows_cache_trans    = False,
                       seed                  = None, 
                       joins_metadata        = True,
                       mpi_comm              = None,):
        # Unpack parameters...
        self.size_sample           = size_sample
        self.size_sample_per_class = size_sample_per_class
        self.dataset_list          = dataset_list
        self.trans                 = trans
        self.allows_cache_trans    = allows_cache_trans
        self.joins_metadata        = joins_metadata
        self.seed                  = seed
        self.mpi_comm              = mpi_comm

        # Set up mpi...
        if self.mpi_comm is not None:
            self.mpi_size     = self.mpi_comm.Get_size()    # num of processors
            self.mpi_rank     = self.mpi_comm.Get_rank()
            self.mpi_data_tag = 11

        # Set seed for data spliting...
        if seed is not None:
            set_seed(seed)

        self.random_state_cache_dict = {}
        self.dataset_cache_dict = {}

        # Fetch all metadata...
        self.metadata_list = [ metadata for _, _, metadata in self.dataset_list ]
        self.label_list    = [ label for _, label, _ in self.dataset_list ]

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


    def cache_dataset(self, idx_list = []):
        if not len(idx_list): idx_list = range(self.size_sample)
        for idx in idx_list:
            if idx in self.dataset_cache_dict: continue

            print(f"Cacheing data point {idx}...")

            img, label, metadata = self.get_data(idx)
            self.dataset_cache_dict[idx] = (img, label, metadata)

        return None


    def mpi_cache_dataset(self):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        # Import chunking method...
        from ..utils import split_list_into_chunk

        # Get the MPI metadata...
        mpi_comm     = self.mpi_comm
        mpi_size     = self.mpi_size
        mpi_rank     = self.mpi_rank
        mpi_data_tag = self.mpi_data_tag

        # If subset is not give, then go through the whole set...
        idx_list = range(self.size_sample)

        # Split the workload...
        idx_list_in_chunk = split_list_into_chunk(idx_list, max_num_chunk = mpi_size)

        # Process chunk by each worker...
        # No need to sync the dataset_cache_dict across workers
        if mpi_rank != 0:
            if mpi_rank < len(idx_list_in_chunk):
                idx_list_per_worker = idx_list_in_chunk[mpi_rank]
                self.dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

            mpi_comm.send(self.dataset_cache_dict, dest = 0, tag = mpi_data_tag)

        if mpi_rank == 0:
            idx_list_per_worker = idx_list_in_chunk[mpi_rank]
            self.dataset_cache_dict = self._mpi_cache_data_per_rank(idx_list_per_worker)

            for i in range(1, mpi_size, 1):
                dataset_cache_dict = mpi_comm.recv(source = i, tag = mpi_data_tag)

                self.dataset_cache_dict.update(dataset_cache_dict)

        return None


    def _mpi_cache_data_per_rank(self, idx_list):
        ''' Cache image in the seq_random_list unless a subset is specified
            using MPI.
        '''
        dataset_cache_dict = {}
        for idx in idx_list:
            # Skip those have been recorded...
            if idx in dataset_cache_dict: continue

            print(f"Cacheing data point {idx}...")

            img, label, metadata = self.get_data(idx)
            dataset_cache_dict[idx] = (img, label, metadata)

        return dataset_cache_dict


    def get_data(self, idx):
        '''
        Acutallay access the source data and apply the tranformation.
        '''

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

        return img, label, metadata


    def __getitem__(self, idx):
        # Lazy load a cached image if it exists...
        img, label, metadata = self.dataset_cache_dict[idx] if idx in self.dataset_cache_dict \
                                                            else self.get_data(idx)

        # Normalize input image...
        img_mean = np.mean(img)
        img_std  = np.std(img)
        img      = (img - img_mean) / img_std

        if self.joins_metadata: metadata = ' '.join(metadata)

        return img[None,], label, metadata


    def _form_online_set(self):
        """ 
        Creating `size_sample` simple set that consists of one image only. 
        """
        # Select two list of random labels following uniform distribution...
        # For a single image
        size_sample = self.size_sample
        label_online_list  = random.choices(self.labels, k = size_sample)

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
        for label in label_online_list:
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
            label = self.label_list[idx]

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
            metadata = self.metadata_list[i]
            res += [' '.join(metadata), ]

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


