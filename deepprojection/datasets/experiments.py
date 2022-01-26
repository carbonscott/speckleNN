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
import json
import csv
import os


class SPIImgDataset(Dataset):
    """
    SPI images are collected from multiple datasets specified in the input csv
    file. All images are organized in a plain list.  
    """

    def __init__(self, fl_csv):
        """
        Args:
            fl_csv (string) : CSV file of datasets.
        """
        self.dataset_dict  = {}
        self.imglabel_list = []
        self.psana_imgreader_dict = {}

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
                imglabel_fileparser         = ImgLabelFileParser(exp, run, drc_root)
                self.dataset_dict[basename] = imglabel_fileparser.imglabel_dict

        # Enumerate each image from all datasets
        for dataset_id, dataset_content in self.dataset_dict.items():
            # Get the exp and run
            ## exp, run = dataset_id.split(".")
            exp, run = dataset_id

            for event_num, label in dataset_content.items():
                self.imglabel_list.append( (exp, run, int(event_num), int(label)) )

        return None


    def __len__(self):
        return len(self.imglabel_list)


    def __getitem__(self, idx):
        exp, run, event_num, label = self.imglabel_list[idx]

        ## print(f"Loading image {exp}.{run}.{event_num}...")

        basename = (exp, run)
        img = self.psana_imgreader_dict[basename].get(int(event_num))

        return img.reshape(-1), int(label)


    def get_imagesize(self, idx):
        exp, run, event_num, label = self.imglabel_list[idx]

        ## print(f"Loading image {exp}.{run}.{event_num}...")

        basename = (exp, run)
        img = self.psana_imgreader_dict[basename].get(int(event_num))

        return img.shape




class SiameseDataset(SPIImgDataset):
    """
    Siamese requires an input of three images at a time, namely anchor,
    positive, and negative.  This dataset will create such triplet
    automatically by randomly choosing an anchor followed up by randomly
    selecting a positive and negative, respectively.
    """

    def __init__(self, fl_csv, size_sample, debug = False):
        super().__init__(fl_csv)

        self.num_stockimgs = len(self.imglabel_list)
        self.size_sample   = size_sample
        self.debug         = debug

        # Create a lookup table for locating the sequence number (seqi) based on a label
        self.label_seqi_dict = {}
        for seqi, (_, _, _, label) in enumerate(self.imglabel_list):
            if not label in self.label_seqi_dict: self.label_seqi_dict[label] = [seqi]
            else: self.label_seqi_dict[label].append(seqi)

        self.triplets = self._form_tripets()

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

        if self.debug: 
            # Append (exp, run, event_num, label) to the result
            for i in (id_anchor, id_pos, id_neg): 
                title = [ str(j) for j in self.imglabel_list[i] ]
                res += (' '.join(title), )

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
            _, _, _, label_anchor = self.imglabel_list[id_anchor]

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




class ImgLabelFileParser:
    """
    It parses a label file associated with a run in an experiment.  The label 
    file, a json file of event number and labels, should be generated by
    psocake.  This parser numerically sorts the event number and assign a
    zero-based index to each event number.  This is implemented primarily for
    complying with PyTorch DataLoader.  
    """

    def __init__(self, exp, run, drc_root):
        self.exp                = exp
        self.run                = run
        self.drc_root           = drc_root
        self.path_labelfile     = ""
        self.imglabel_dict = {}

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
