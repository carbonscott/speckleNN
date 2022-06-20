#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load PyTorch
import torch
from torch.utils.data import Dataset

# Load LCLS data management and LCLS data access (through detector) module
import psana

# Load misc modules
import numpy as np
import random
import csv
import os

import logging

from deepprojection.utils import downsample, set_seed

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''
    NOLABEL    = '-1'
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
        fl_csv          = getattr(config, 'fl_csv'        , None)
        exclude_labels  = getattr(config, 'exclude_labels', None)
        self.isflat     = getattr(config, 'isflat'        , None)
        self.mode       = getattr(config, 'mode'          , None)
        self.istrain    = getattr(config, 'istrain'       , None)
        self.frac_train = getattr(config, 'frac_train'    , None)    # Proportion/Fraction of training examples
        self.seed       = getattr(config, 'seed'          , None)
        self.trans      = getattr(config, 'trans'         , None)

        self._dataset_dict        = {}
        self.psana_imgreader_dict = {}
        self.imglabel_orig_list   = []

        # Set the seed...
        # Debatable whether seed should be set in the dataset or in the running code
        if not self.seed is None: set_seed(self.seed)

        # Read csv file of datasets...
        with open(fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            # Skip the header...
            next(lines)

            # Read each line/dataset...
            for line in lines:
                # Fetch metadata of a dataset 
                exp, run, mode, detector_name, drc_label = line

                # Form a minimal basename to describe a dataset...
                basename = (exp, run)

                # Initiate image accessing layer...
                psana_imgreader = PsanaImg(exp, run, mode, detector_name)
                self.psana_imgreader_dict[basename] = psana_imgreader

                # Obtain image labels from this dataset...
                imglabel_fileparser = ImgLabelFileParser(exp, run, drc_label, exclude_labels)

                # Parse labels in the label file if it exists???
                self._dataset_dict[basename] = imglabel_fileparser.imglabel_dict


        # Enumerate each labeled image from all datasets...
        for dataset_id, dataset_content in self._dataset_dict.items():
            # Get the exp and run
            exp, run = dataset_id

            for event_num, label in dataset_content.items():
                self.imglabel_orig_list.append( (exp, run, f"{event_num:>6s}", label) )

        # Split the original image list into training set and test set...
        num_list  = len(self.imglabel_orig_list)
        num_train = int(self.frac_train * num_list)

        # Get training examples...
        imglabel_train_list = random.sample(self.imglabel_orig_list, num_train)

        # Get test examples...
        imglabel_test_list = set(self.imglabel_orig_list) - set(imglabel_train_list)
        imglabel_test_list = sorted(list(imglabel_test_list))

        self.imglabel_list = imglabel_train_list if self.istrain else imglabel_test_list

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def get_img_and_label(self, idx):
        # Read image...
        exp, run, event_num, label = self.imglabel_list[idx]
        basename = (exp, run)
        img = self.psana_imgreader_dict[basename].get(int(event_num), mode = self.mode)

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
        for seqi, (_, _, _, label) in enumerate(self.imglabel_list):
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

        # Read the anchor, pos, neg
        img_anchor, label_anchor = super().__getitem__(id_anchor)
        img_pos, _ = super().__getitem__(id_pos)
        img_neg, _ = super().__getitem__(id_neg)

        res = img_anchor, img_pos, img_neg, label_anchor

        # Append (exp, run, event_num, label) to the result
        for i in (id_anchor, id_pos, id_neg): 
            ## title = [ str(j) for j in self.imglabel_list[i] ]
            title = self.imglabel_list[i]
            res += (' '.join(title), )

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

        # Read the anchor, pos, neg
        img_anchor, label_anchor = super().__getitem__(id_anchor)
        img_second, _ = super().__getitem__(id_second)

        res = img_anchor, img_second, label_anchor

        # Append (exp, run, event_num, label) to the result
        for i in (id_anchor, id_second): 
            ## title = [ str(j) for j in self.imglabel_list[i] ]
            title = self.imglabel_list[i]
            res += (' '.join(title), )

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

        # Append (exp, run, event_num, label) to the result
        for i in query_tuple: 
            title = self.imglabel_list[i]
            res += [' '.join(title), ]

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




class ImgLabelFileParser:
    """
    It parses a label file associated with a run in an experiment.  The label 
    file, a csv file of event number and labels, should be generated by
    psocake.  This parser numerically sorts the event number and assign a
    zero-based index to each event number.  This is implemented primarily for
    complying with PyTorch DataLoader.  

    The type of a lable is always string.  
    """

    def __init__(self, exp, run, drc_label, exclude_labels = ()):
        self.exp            = exp
        self.run            = run
        self.drc_label      = drc_label
        self.path_labelfile = ""
        self.imglabel_dict  = {}
        self.exclude_labels = exclude_labels

        # Initialize indexed image label
        self._load_imglabel()

        return None


    def __getitem__(self, idx):
        return self.indexed_imglabel_dict[idx]


    def _locate_labelfile(self):
        # Basename of a label file
        basename = f"{self.exp}_{int(self.run):04d}"

        # Locate the path to label file
        fl_label = f"{basename}.label.csv"
        path_labelfile = os.path.join(self.drc_label, fl_label)

        return path_labelfile


    def _load_imglabel(self):
        # Load path to the label file
        self.path_labelfile = self._locate_labelfile()

        # Read, sort and index labels
        imglabel_dict = {}
        if os.path.exists(self.path_labelfile):
            # Read csv file of datasets
            with open(self.path_labelfile, 'r') as fh:
                lines = csv.reader(fh)

                # Skip the header
                next(lines)

                # Read each line/dataset
                for line in lines:
                    # Fetch metadata of a dataset 
                    _, _, event_num, label = line
                    imglabel_dict[event_num] = label

            # Exclude some labels
            imglabel_dict = { k:v for k, v in imglabel_dict.items() if not v in self.exclude_labels }

            # Sort label
            self.imglabel_dict = dict( sorted( imglabel_dict.items(), key = lambda x:int(x[0]) ) )

        return None




class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)


    def get(self, event_num, mode = "image"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only three modes are supported...
        assert mode in ("raw", "image", "calib"), f"Mode {mode} is not allowed!!!  Only 'raw' or 'image' are supported."

        # Fetch image data based on timestamp from detector...
        read = { "image" : self.detector.image, }
        img = read[mode](event)

        return img




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

        # Append to the result...
        exp, run, event_num, label = self.imglabel_list[id_single]
        title = f"{exp} {run} {event_num:>06d} {label}"
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




class SequentialSet(Siamese):
    """
    Sequential set feeds one example to model at a time.  The purpose is simply to
    encode each example in imglabel_list.  
    """

    def __init__(self, config):
        super().__init__(config)

        # Create a lookup table for locating the sequence number (seqi) based on a label...
        self.label_seqi_orig_dict = {}
        for seqi, (_, _, _, label) in enumerate(self.imglabel_orig_list):
            # Keep track of label and its seqi
            if not label in self.label_seqi_orig_dict: self.label_seqi_orig_dict[label] = [seqi]
            else                                     : self.label_seqi_orig_dict[label].append(seqi)

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def __getitem__(self, idx):
        # Read the single image...
        img, _ = super().__getitem__(idx)
        res = (img, )

        # Append metadata to the result list...
        exp, run, event_num, label = self.imglabel_orig_list[idx]
        title = f"{exp} {run} {int(event_num):>06d} {label}"
        res += (title, )

        return res




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

        # Append to the result...
        exp, run, event_num, label = self.imglabel_list[id_single]
        title = f"{exp} {run} {int(event_num):>06d} {label}"
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


    def report(self):
        # Log the number of images for each label...
        logger.info("___/ List of entries in dataset \___")

        count_per_label_dict = {}
        for idx in self.online_set:
            exp, run, event_num, label = self.imglabel_list[idx]
            logger.info(f"ENTRIES - {exp:12s} {int(run):04d} {int(event_num):06d} {label:2s}")

            if not label in count_per_label_dict: count_per_label_dict[label]  = 1
            else                                : count_per_label_dict[label] += 1

        for label, count in count_per_label_dict.items():
            logger.info(f"COUNTS - label {label} : {count}")
