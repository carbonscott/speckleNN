#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Load PyTorch
import torch
from torch.utils.data import Dataset
import skimage.transform

# Load LCLS data management and LCLS data access (through detector) module
import psana

# Load misc modules
import numpy as np
import random
import json
import csv
import os

import logging

from deepprojection.utils import downsample

logger = logging.getLogger(__name__)

class ConfigDataset:
    ''' Biolerplate code to config dataset classs'''
    NOHIT    = '0'
    SINGLE   = '1'
    MULTI    = '2'
    UNKNOWN  = '3'
    NEEDHELP = '4'

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Dataset \___")
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class SPIImgDataset(Dataset):
    """
    SPI images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  
    """

    def __init__(self, config):
        """
        Args:
            fl_csv (string) : CSV file of datasets.
        """
        fl_csv         = config.fl_csv
        exclude_labels = config.exclude_labels
        self.resize    = config.resize
        self.isflat    = config.isflat

        self._dataset_dict        = {}
        self.psana_imgreader_dict = {}
        self.imglabel_list        = []

        # Read csv file of datasets
        with open(fl_csv, 'r') as fh:
            lines = csv.reader(fh)

            # Skip the header
            next(lines)

            # Read each line/dataset
            for line in lines:
                # Fetch metadata of a dataset 
                exp, run, mode, detector_name, drc_root = line

                # Form a minimal basename to describe a dataset
                ## basename = f"{exp}.{run}"
                basename = (exp, run)

                # Initiate image accessing layer
                self.psana_imgreader_dict[basename] = PsanaImg(exp, run, mode, detector_name)

                # Obtain image labels from this dataset
                imglabel_fileparser         = ImgLabelFileParser(exp, run, drc_root, exclude_labels)
                self._dataset_dict[basename] = imglabel_fileparser.imglabel_dict

        # Enumerate each image from all datasets
        for dataset_id, dataset_content in self._dataset_dict.items():
            # Get the exp and run
            ## exp, run = dataset_id.split(".")
            exp, run = dataset_id

            for event_num, label in dataset_content.items():
                self.imglabel_list.append( (exp, run, f"{event_num:>6s}", label) )

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def get_img_and_label(self, idx):
        # Read image...
        exp, run, event_num, label = self.imglabel_list[idx]
        basename = (exp, run)
        img = self.psana_imgreader_dict[basename].get(int(event_num))

        # Resize images...
        if self.resize:
            bin_row, bin_col = self.resize
            img = downsample(img, bin_row, bin_col, mask = None)

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
        self.labels = list(set([ i[-1] for i in self.imglabel_list ]))

        # Log the number of images for each label...
        logger.info("___/ Dataset statistics \___")
        for label in self.labels:
            num_img = len(label_seqi_dict[label])
            logger.info(f"KV - {label:16s} : {num_img}")

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
        super().__init__(config)

        label_seqi_dict = self.label_seqi_dict

        ## self.size_sample = getattr(config, 'size_sample')

        ## # Create a lookup table for locating the sequence number (seqi) based on a label...
        ## label_seqi_dict = {}
        ## for seqi, (_, _, _, label) in enumerate(self.imglabel_list):
        ##     # Keep track of label and its seqi
        ##     if not label in label_seqi_dict: label_seqi_dict[label] = [seqi]
        ##     else                           : label_seqi_dict[label].append(seqi)

        ## # Consolidate labels in the dataset...
        ## self.labels = list(set([ i[-1] for i in self.imglabel_list ]))

        ## # Log the number of images for each label...
        ## logger.info("___/ Dataset statistics \___")
        ## for label in self.labels:
        ##     num_img = len(label_seqi_dict[label])
        ##     logger.info(f"KV - {label:16s} : {num_img}")

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




class ImgLabelFileParser:
    """
    It parses a label file associated with a run in an experiment.  The label 
    file, a json file of event number and labels, should be generated by
    psocake.  This parser numerically sorts the event number and assign a
    zero-based index to each event number.  This is implemented primarily for
    complying with PyTorch DataLoader.  

    The type of a lable is always string.  
    """

    def __init__(self, exp, run, drc_root, exclude_labels = ()):
        self.exp            = exp
        self.run            = run
        self.drc_root       = drc_root
        self.path_labelfile = ""
        self.imglabel_dict  = {}
        self.exclude_labels = exclude_labels

        # Initialize indexed image label
        self._load_imglabel()

        return None


    def __getitem__(self, idx):
        return self.indexed_imglabel_dict[idx]


    def _locate_labelfile(self):
        # Get the username
        username = os.environ.get("USER")

        # The prefix directory to find label file
        drc_run     = f"r{int(self.run):04d}"
        drc_psocake = os.path.join(self.exp, username, 'psocake', drc_run)

        # Basename of a label file
        basename = f"{self.exp}_{int(self.run):04d}"

        # Locate the path to label file
        fl_json = f"{basename}.label.json"
        path_labelfile = os.path.join(self.drc_root, drc_psocake, fl_json)

        return path_labelfile


    def _load_imglabel(self):
        # Load path to the label file
        self.path_labelfile = self._locate_labelfile()

        # Read, sort and index labels
        if os.path.exists(self.path_labelfile):
            # Read label
            with open(self.path_labelfile, 'r') as fh:
                imglabel_dict = json.load(fh)

            # Exclude some labels
            imglabel_dict = { k:v for k, v in imglabel_dict.items() if not v in self.exclude_labels }

            # Sort label
            self.imglabel_dict = dict( sorted( imglabel_dict.items(), key = lambda x:int(x[0]) ) )


        else:
            print(f"File doesn't exist!!! Missing {self.path_labelfile}.")

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


    def get(self, event_num):
        # Fetch the timestamp according to event number
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector
        img = self.detector.image(event)

        return img
