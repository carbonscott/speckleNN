#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This program uses `SpeckleNN` to predict query labels for a test set and
computes a confusion matrix for evaluating the prediction performance.

User inputs
-----------

timestamp : str
    Specifies a model checkpoint for the prediction along with epoch.

epoch : int
    Specifies a model checkpoint for the prediction along with timestamp.


num_max_support : int
    Specifies the number of support examples for each category in few-shot
    learning.

tag : str
    Any text that user wants to include in the filename of the output.  The
    first character in the tag should be '.'.

    Example: '.bucket0'

scaling_exponent_list : list or np.ndarray of float, shape (N,)
    A list of scaling exponents used in scaling the fluence of diffraction
    images.  The scaling preceeds all preprocessing steps.  This variable can
    also be a numpy array.

path_pdb_list : str
    A path to the file that contains a list of PDB entries to use in the model
    prediction.
'''

import os
import logging
import torch
import socket
import pickle
import tqdm
import random
import numpy as np

from deepprojection.datasets.lite    import SPIOnlineDataset
from deepprojection.model            import OnlineTripletSiameseModel, ConfigSiameseModel
from deepprojection.encoders.convnet import FewShotModel, ConfigEncoder
from deepprojection.utils            import EpochManager, MetaLog, init_logger, split_dataset, set_seed, ConfusionMatrix

from image_preprocess_faulty_sq_for_validate import DatasetPreprocess

# [[[ CONSTANTS ]]]
PATH_INPUT  = "."
PATH_OUTPUT = "confusion_matrix"

# [[[ SEED ]]]
seed = 0
set_seed(seed)


# [[[ CONFIG ]]]
timestamp             = "2023_0101_0856_44"
epoch                 = 71
num_max_support       = 5
tag                   = f".test"
fl_chkpt              = f"{timestamp}.epoch={epoch}.chkpt"
size_sample_query     = 1000
frac_support          = 0.4
size_batch            = 100
trans                 = None
alpha                 = 0.02
scaling_exponent_list = np.linspace(-2, 2, 101)
scaling_exponent_list = np.array([0.0, 2.0])
## path_pdb_list         = 'skopi/h5s_mini.sq.test.dat'
## path_pdb_list         = 'skopi/h5s_mini.sq.test.corrected.dat'
## path_pdb_list         = 'skopi/h5s_mini.sq.test.bucket18.dat'
## path_pdb_list         = 'skopi/h5s_mini.sq.test.bucket01.dat'
path_pdb_list         = 'skopi/h5s_mini.sq.test.bucket02.dat'
## path_pdb_list         = 'skopi/h5s_mini.sq.test.bucket17.dat'

# Initialize a log file...
init_logger(log_name = 'validate.query.test', timestamp = timestamp, returns_timestamp = False, saves_log = False)

# Load PDBs...
pdb_list = open(path_pdb_list).readlines()
pdb_list = [ pdb.strip() for pdb in pdb_list ]

# Save performance by pdb...
pdb_to_perf_dict = {}
for enum_pdb, pdb in enumerate(pdb_list):
    pdb_to_perf_dict[pdb] = []

    # [[[ DATASET ]]]
    # Set up parameters for an experiment...
    drc_dataset  = 'fastdata.h5/'
    fl_dataset   = f'{pdb}.relabel.pickle'
    path_dataset = os.path.join(drc_dataset, fl_dataset)

    # Load raw data...
    with open(path_dataset, 'rb') as fh:
        dataset_list = pickle.load(fh)

    # Scan through a list of shot to shot fluc setting...
    for enum_photon, scaling_exponent in enumerate(scaling_exponent_list):
        photon_scale = 10 ** scaling_exponent

        # Increase the photon intensities...
        dataset_rescale_list = [ (img * photon_scale, label, metadata) for (img, label, metadata) in dataset_list ]

        # Set seed for reproducibility
        set_seed(seed)

        # Split data into two -- support set and query set...
        data_support, data_query = split_dataset(dataset_rescale_list, frac_support, seed = None)

        # Fetch all hit labels...
        hit_list = list(set( [ hit for _, (pdb, hit), _ in data_support ] ))

        # Form support set...
        support_hit_to_idx_dict = { hit : [] for hit in hit_list }
        for enum_data, (img, label, metadata) in enumerate(data_support):
            _, hit = label
            support_hit_to_idx_dict[hit].append(enum_data)

        for hit, idx_support in support_hit_to_idx_dict.items():
            if len(support_hit_to_idx_dict[hit]) > num_max_support:
                support_hit_to_idx_dict[hit] = random.sample(support_hit_to_idx_dict[hit], k = num_max_support)

        # Form query dataset...
        dataset_query = SPIOnlineDataset( dataset_list       = data_query,
                                          prints_cache_state = False,
                                          size_sample        = size_sample_query,
                                          joins_metadata     = False,
                                          trans              = trans, )

        if enum_pdb == 0 and enum_photon == 0:
            # [[[ Preprocess dataset ]]]
            # Data preprocessing can be lengthy and defined in dataset_preprocess.py
            img_orig            = dataset_query[0][0][0]   # idx, fetch img
            dataset_preproc     = DatasetPreprocess(img_orig)
            trans               = dataset_preproc.config_trans()
            dataset_query.trans = trans
            img_trans           = dataset_query[0][0][0]

            # [[[ IMAGE ENCODER ]]]
            # Config the encoder...
            dim_emb        = 128
            size_y, size_x = img_trans.shape[-2:]
            config_encoder = ConfigEncoder( dim_emb = dim_emb,
                                            size_y  = size_y,
                                            size_x  = size_x,
                                            isbias  = True )
            encoder = FewShotModel(config_encoder)

            # [[[ DEVICE ]]]
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

            # [[[ MODEL ]]]
            # Config the model...
            config_siamese = ConfigSiameseModel( alpha = alpha, encoder = encoder, )
            model = OnlineTripletSiameseModel(config_siamese)
            model.init_params(fl_chkpt = fl_chkpt)
            model.to(device = device)

        # Set seed for reproducibility
        set_seed(seed)

        # Cache it otherwise it changes each time it is called...
        dataset_query.cache_dataset()

        # [[[ EMBEDDING (SUPPORT) ]]]
        support_batch_emb_dict = { hit : None for hit in hit_list }
        for hit in hit_list:
            support_idx_list    = support_hit_to_idx_dict[hit]
            num_example_support = len(support_idx_list)
            for enum_support_idx, support_idx in enumerate(support_idx_list):
                # Fetch data from support...
                img = data_support[support_idx][0]
                img = trans(img)
                img = img[None,]

                # Normalize the image...
                img = (img - img.mean()) / img.std()

                # Preallocate tensor...
                if enum_support_idx == 0:
                    size_c, size_y, size_x = img.shape
                    batch_img = torch.zeros((num_example_support, size_c, size_y, size_x))

                # Save image as tensor...
                batch_img[enum_support_idx] = torch.tensor(img)

            with torch.no_grad():
                batch_img = batch_img.to(device = device)
                support_batch_emb_dict[hit] = model.encoder.encode(batch_img)

        # [[[ EMBEDDING (QUERY) ]]]
        num_test       = len(dataset_query)
        query_idx_list = range(num_test)
        for enum_query_idx, i in enumerate(query_idx_list):
            # Fetch data from query
            img = dataset_query[i][0]

            # Preallocate tensor...
            if enum_query_idx == 0:
                size_c, size_y, size_x = img.shape
                batch_img = torch.zeros((num_test, size_c, size_y, size_x))

            # Save image as tensor...
            batch_img[enum_query_idx] = torch.tensor(img)

        with torch.no_grad():
            batch_img = batch_img.to(device = device)
            query_batch_emb = model.encoder.encode(batch_img)


        # [[[ METRIC ]]]
        diff_query_support_dict = {}
        for hit in hit_list:
            # Q: number of query examples.
            # S: number of support examples
            # E: dimension of an embedding
            # diff_query_support_dict[hit]: Q x S x E
            # query_batch_emb[:, None]    : Q x 1 x E
            # support_batch_emb_dict[hit] :     S x E
            diff_query_support_dict[hit] = query_batch_emb[:,None] - support_batch_emb_dict[hit]

        dist_dict = {}
        for hit in hit_list:
            # Q: number of query examples.
            # S: number of support examples
            # dist_dict[hit]: Q x S
            dist_dict[hit] = torch.sum(diff_query_support_dict[hit] * diff_query_support_dict[hit], dim = -1)

        # Use enumeration as an intermediate to obtain predicted hits...
        enum_to_hit_dict = {}

        # Encode hit type with enum
        # enum 0 : hit 1
        # enum 1 : hit 2
        for enum_hit, hit in enumerate(hit_list):
            enum_to_hit_dict[enum_hit] = hit

            # Fetch the values and indices of the closet support for this hit type for the query...
            mean_support_val = dist_dict[hit].mean(dim = -1)
            if enum_hit == 0:
                # H: number of hit types (single vs multi)
                # N: number of examples
                # mean_support_tensor: H x N
                mean_support_tensor = torch.zeros((len(hit_list), *mean_support_val.shape))

            mean_support_tensor[enum_hit] = mean_support_val

        # Obtain the predicted hit...
        # Obtain the min among examples across all hit type (dim = 0)
        # [1] is to pick the indices from the result of a torch.min
        pred_hit_as_enum_list = mean_support_tensor.min(dim = 0)[1]

        # Obtain the predicted hit for each input example
        pred_hit_list = [ enum_to_hit_dict[enum.item()] for enum in pred_hit_as_enum_list ]

        # Obtain the real hit...
        real_hit_list = [ dataset_query[idx][1][1] for idx in query_idx_list ]

        # New container to store validation result (thus res_dict) for each label...
        res_dict = {}
        for hit in hit_list: res_dict[hit] = { i : [] for i in hit_list }

        for pred_hit, real_hit in zip(pred_hit_list, real_hit_list):
            res_dict[pred_hit][real_hit].append( None )

        # Characterize performance by the confusion matrix measured at each scaling point...
        perf = [scaling_exponent, res_dict]
        pdb_to_perf_dict[pdb].append(perf)

        acc = ConfusionMatrix(res_dict).get_metrics(1)[0]
        print(f"Working on {pdb}, {scaling_exponent}, {acc}...")


    fl_pdb_to_perf_dict = f'{timestamp}.epoch_{epoch}.support_{num_max_support}.mean_dist{tag}.pickle'
    path_pdb_to_perf_dict = os.path.join(PATH_OUTPUT, fl_pdb_to_perf_dict)
    with open(path_pdb_to_perf_dict, 'wb') as handle:
        pickle.dump(pdb_to_perf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
