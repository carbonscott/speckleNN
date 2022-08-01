#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
mpiexec -n 11 python mpi.train.online.simulated.square_detector.py
'''

from mpi4py import MPI

import os
import logging
import torch
from deepprojection.datasets.simulated_square_detector import OnlineDataset      , ConfigDataset
from deepprojection.model                              import OnlineSiameseModel , ConfigSiameseModel
from deepprojection.trainer                            import OnlineTrainer      , ConfigTrainer
from deepprojection.validator                          import OnlineLossValidator, ConfigValidator
from deepprojection.encoders.convnet                   import Hirotaka0122       , ConfigEncoder
## from deepprojection.encoders.convnet                   import Hirotaka0122Plus   , ConfigEncoder
from deepprojection.utils                              import EpochManager       , MetaLog
from simulated_square_detector_preprocess              import DatasetPreprocess
from datetime import datetime
import socket

def chunk_list(input_list, num_chunk = 2):

    chunk_size = len(input_list) // num_chunk + 1

    chunked_list = []
    for idx_chunk in range(num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_chunk == num_chunk - 1: idx_e = len(input_list)
        seg = input_list[idx_b : idx_e]
        chunked_list.append(seg)

    return chunked_list


# ___/ MAIN \___
# [[[ MPI HEADER ]]]
# Set up MPI...
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()    # num of processors
mpi_rank = mpi_comm.Get_rank()
mpi_data_tag  = 11

# Is it a continued training???
timestamp_prev = None
## timestamp_prev = "2022_0717_1937_09"

# [[[ PARAMETERS ]]]
# Set up MPI...
# Set up parameters for an experiment...
## fl_csv                = "simulated.square_detector.datasets.6Q5U.csv"

fl_csv                = "simulated.square_detector.datasets.pdb_sampled.80.csv"
size_sample_train     = 200000
size_sample_validate  = 80000
frac_train            = 0.7

## fl_csv                = "simulated.square_detector.datasets.pdb_sampled.10.csv"
## size_sample_train     = 50000
## size_sample_validate  = 10000
## frac_train            = 0.7

## fl_csv                = "simulated.square_detector.datasets.pdb_sampled.50.csv"
## size_sample_train     = 200000
## size_sample_validate  = 80000
## frac_train            = 0.7

dim_emb = 128
alpha   = 0.2
sigma   = 0.15 * 1    # ...Define Gaussian noise level

size_sample_per_class = None
size_batch            = 1000
frac_validate         = None
dataset_usage         = 'train'
online_shuffle        = True
lr                    = 1e-3
seed                  = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)     : {size_sample_train}
            Sample size (validate)  : {size_sample_validate}
            Sample size (per class) : {size_sample_per_class}
            Batch  size             : {size_batch}
            Alpha                   : {alpha}
            Online shuffle          : {online_shuffle}
            lr                      : {lr}
            Continued from???       : {timestamp_prev}

            """

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y_%m%d_%H%M_%S")

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_log         = f"{timestamp}.train.log"
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


# [[[ DATASET ]]]
# Config the dataset...
exclude_labels = [ ConfigDataset.UNKNOWN, ConfigDataset.NEEDHELP, ConfigDataset.NOHIT, ConfigDataset.BACKGROUND ]
config_dataset = ConfigDataset( fl_csv                = fl_csv,
                                size_sample           = size_sample_train, 
                                seed                  = seed,
                                isflat                = False,
                                frac_train            = frac_train,
                                frac_validate         = frac_validate,
                                size_sample_per_class = size_sample_per_class,
                                dataset_usage         = dataset_usage,
                                trans                 = None,
                                exclude_labels        = exclude_labels, )

# Define training set...
dataset_train = OnlineDataset(config_dataset)

# Preprocess dataset...
# Data preprocessing can be lengthy and defined in dataset_preprocess.py
img_orig            = dataset_train[0][0][0]
dataset_preproc     = DatasetPreprocess(img_orig, sigma = sigma)
trans               = dataset_preproc.config_trans()
dataset_train.trans = trans
img_trans           = dataset_train[0][0][0]

# Split MPI work load (training)...
train_idx_list         = list(set(dataset_train.online_set))
train_chunked_idx_list = chunk_list(train_idx_list, mpi_size)

# Load images by each worker...
if mpi_rank != 0:
    train_idx_list_by_rank = train_chunked_idx_list[mpi_rank]
    dataset_train.cache_img(train_idx_list_by_rank)

    mpi_comm.send(dataset_train.imglabel_cache_dict, dest = 0, tag = mpi_data_tag)

if mpi_rank == 0:
    train_idx_list_by_rank = train_chunked_idx_list[mpi_rank]
    dataset_train.cache_img(train_idx_list_by_rank)

    # Combine data from each worker...
    # Data transmition is synced at recv stage as it blocks any following
    # operations until all recv is complete.
    for i in range(1, mpi_size, 1):
        imglabel_cache_dict_by_rank = mpi_comm.recv(source = i, tag = mpi_data_tag)
        dataset_train.imglabel_cache_dict.update(imglabel_cache_dict_by_rank)

    dataset_train.report()

# Report training set...
config_dataset.trans = trans
config_dataset.report()

# Define validation set...
config_dataset.size_sample           = size_sample_validate
config_dataset.dataset_usage         = 'validate'
config_dataset.size_sample_per_class = None
config_dataset.report()
dataset_validate = OnlineDataset(config_dataset)

# Split MPI work load (validation)...
validate_idx_list         = list(set(dataset_validate.online_set))
validate_chunked_idx_list = chunk_list(validate_idx_list, mpi_size)

# Load images by each worker...
if mpi_rank != 0:
    validate_idx_list_by_rank = validate_chunked_idx_list[mpi_rank]
    dataset_validate.cache_img(validate_idx_list_by_rank)

    mpi_comm.send(dataset_validate.imglabel_cache_dict, dest = 0, tag = mpi_data_tag)

if mpi_rank == 0:
    validate_idx_list_by_rank = validate_chunked_idx_list[mpi_rank]
    dataset_validate.cache_img(validate_idx_list_by_rank)

    # Combine data from each worker...
    for i in range(1, mpi_size, 1):
        imglabel_cache_dict_by_rank = mpi_comm.recv(source = i, tag = mpi_data_tag)
        dataset_validate.imglabel_cache_dict.update(imglabel_cache_dict_by_rank)

    # Finalize MPI...
    MPI.Finalize()

    # ___/ MODEL TRAINING/VALIDATION \___
    # [[[ IMAGE ENCODER ]]]
    # Config the encoder...
    size_y, size_x = img_trans.shape
    config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                    size_y  = size_y,
                                    size_x  = size_x,
                                    isbias  = True )
    encoder = Hirotaka0122(config_encoder)
    ## encoder = Hirotaka0122Plus(config_encoder)


    # [[[ CHECKPOINT ]]]
    DRCCHKPT         = "chkpts"
    prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
    fl_chkpt         = f"{timestamp}.train.chkpt"
    path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)


    # [[[ MODEL ]]]
    # Config the model...
    config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
    model = OnlineSiameseModel(config_siamese)

    # Initialize weights or reuse weights from a timestamp...
    def init_weights(module):
        if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
    if timestamp_prev is None: 
        model.apply(init_weights)
    else:
        fl_chkpt_prev = f"{timestamp_prev}.train.chkpt"
        path_chkpt_prev = os.path.join(prefixpath_chkpt, fl_chkpt_prev)
        model.load_state_dict(torch.load(path_chkpt_prev))


    # [[[ TRAINER ]]]
    # Config the trainer...
    config_train = ConfigTrainer( path_chkpt     = path_chkpt,
                                  num_workers    = 0,
                                  batch_size     = size_batch,
                                  pin_memory     = True,
                                  shuffle        = False,
                                  online_shuffle = online_shuffle,
                                  is_logging     = False,
                                  method         = 'random-semi-hard', 
                                  lr             = lr, )

    # Training...
    trainer = OnlineTrainer(model, dataset_train, config_train)


    # [[[ VALIDATOR ]]]
    config_validator = ConfigValidator( path_chkpt     = None,
                                        num_workers    = 0,
                                        batch_size     = size_batch,
                                        pin_memory     = True,
                                        shuffle        = False,
                                        online_shuffle = online_shuffle,
                                        is_logging     = False,
                                        method         = 'random-semi-hard', 
                                        lr             = lr, 
                                        isflat         = False, )  # Conv2d input needs one more dim for batch

    validator = OnlineLossValidator(model, dataset_validate, config_validator)


    # [[[ EPOCH MANAGER ]]]
    max_epochs = 360 * 3
    epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
    epoch_manager.run()
