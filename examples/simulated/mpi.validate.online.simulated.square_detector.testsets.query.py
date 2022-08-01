#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI

import os
import logging
import torch
from deepprojection.datasets.simulated_square_detector import MultiwayQueryset      , ConfigDataset
from deepprojection.model                              import SiameseModelCompare   , ConfigSiameseModel
from deepprojection.validator                          import MultiwayQueryValidator, ConfigValidator
from deepprojection.encoders.convnet                   import Hirotaka0122          , ConfigEncoder
## from deepprojection.encoders.convnet                   import Hirotaka0122Plus      , ConfigEncoder
from deepprojection.datasets                           import transform
from deepprojection.utils                              import MetaLog
from simulated_square_detector_preprocess              import DatasetPreprocess
import itertools
import socket

def chunk_list(input_list, num_chunk = 2):

    chunk_size = len(input_list) // num_chunk + 1

    chunked_list = []
    for idx_chunk in range(num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_chunk == num_chunk - 1: idx_e = len(input_list)
        seg   = input_list[idx_b : idx_e]
        chunked_list.append(seg)

    return chunked_list


# ___/ MAIN \___

# Set up MPI...
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()    # num of processors
mpi_rank = mpi_comm.Get_rank()
mpi_data_tag  = 11


# Create a timestamp to name the log file...
## fl_csv = "simulated.square_detector.datasets.pdb_not_sampled.80.csv"
## timestamp = "2022_0718_2212_13"

## fl_csv = "simulated.square_detector.datasets.pdb_not_sampled.10.csv"
## timestamp = "2022_0718_2219_19"
## timestamp = "2022_0718_2223_30"
## timestamp = "2022_0718_2247_15"

## fl_csv = "simulated.square_detector.datasets.pdb_not_sampled.50.csv"
## timestamp = "2022_0718_2226_20"
## timestamp = "2022_0718_2228_01"
## timestamp = "2022_0718_2241_38"


fl_csv = "simulated.square_detector.datasets.pdb_not_sampled.common.csv"
## timestamp = "2022_0718_2247_15"
## timestamp = "2022_0718_2208_54"
## timestamp = "2022_0718_2241_38"
## timestamp = "2022_0718_2212_13"
## timestamp = "2022_0718_2149_51"
## timestamp = "2022_0718_2226_20"
## timestamp = "2022_0718_2219_19"
## timestamp = "2022_0712_1135_25"
timestamp = "2022_0719_2156_29"
## timestamp = "2022_0719_2159_21"
## timestamp = "2022_0719_2159_53"

# Set up parameters for an experiment...
dim_emb = 128

num_query      = 100000
size_batch     = 1000
frac_train     = 1.0
frac_validate  = None
dataset_usage  = 'train'
lr             = 1e-3
seed           = 0

# Comment this verification...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Sample size    : {num_query}
            Batch  size    : {size_batch}
            lr             : {lr}

            Apply model to completed datasets, e.g. not in training or testing.
            """

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.validate.query.generalizability.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Only log from the main worker...
if mpi_rank == 0:
    # Config logging behaviors
    logging.basicConfig( filename = path_log,
                         filemode = 'w',
                         format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                         datefmt="%m/%d/%Y %H:%M:%S",
                         level=logging.INFO, )
    logger = logging.getLogger(__name__)

    # Create a metalog to the log file, explaining the purpose of this run...
    metalog = MetaLog( comments = comments )
    metalog.report()


# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv            = fl_csv,
                                size_sample       = num_query, 
                                seed              = seed,
                                isflat            = False,
                                frac_train        = frac_train,
                                frac_validate     = frac_validate,
                                dataset_usage     = dataset_usage,
                                exclude_labels    = exclude_labels, )

# Define validation set...
dataset_validate = MultiwayQueryset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig               = dataset_validate[0][0][0]
dataset_preproc        = DatasetPreprocess(img_orig)
trans                  = dataset_preproc.config_trans()
dataset_validate.trans = trans
img_trans              = dataset_validate[0][0][0]

# Split MPI work load...
idx_list = list(set(itertools.chain(*dataset_validate.queryset)))
chunked_idx_list = chunk_list(idx_list, mpi_size)

# Load images by each worker...
if mpi_rank != 0:
    idx_list_by_rank = chunked_idx_list[mpi_rank]
    dataset_validate.cache_img(idx_list_by_rank)

    mpi_comm.send(dataset_validate.imglabel_cache_dict, dest = 0, tag = mpi_data_tag)

if mpi_rank == 0:
    # Main worker does the data loading...
    idx_list_by_rank = chunked_idx_list[mpi_rank]
    dataset_validate.cache_img(idx_list_by_rank)

    # Combine data from each worker...
    for i in range(1, mpi_size, 1):
        imglabel_cache_dict_by_rank = mpi_comm.recv(source = i, tag = mpi_data_tag)
        dataset_validate.imglabel_cache_dict.update(imglabel_cache_dict_by_rank)

    # Finalize MPI...
    MPI.Finalize()

    # Define validation set...
    config_dataset.trans = trans
    config_dataset.report()

    # Fetch checkpoint directory...
    DRCCHKPT = "chkpts"
    prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)

    # Config the encoder...
    size_y, size_x = img_trans.shape
    config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                    size_y  = size_y,
                                    size_x  = size_x,
                                    isbias  = True )
    encoder = Hirotaka0122(config_encoder)
    ## encoder = Hirotaka0122Plus(config_encoder)

    # Set up the model
    config_siamese = ConfigSiameseModel( encoder = encoder, )
    model = SiameseModelCompare(config_siamese)

    # Read chkpt from a trainig
    fl_chkpt = f"{timestamp}.train.chkpt"
    path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)
    config_validator = ConfigValidator( path_chkpt  = path_chkpt,
                                        num_workers = 1,
                                        batch_size  = size_batch,
                                        pin_memory  = True,
                                        shuffle     = False,
                                        isflat      = False,
                                        lr          = lr, )

    validator = MultiwayQueryValidator(model, dataset_validate, config_validator)
    validator.validate()
