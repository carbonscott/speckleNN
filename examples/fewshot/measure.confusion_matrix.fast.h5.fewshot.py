#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


# [[[ SEED ]]]
seed = 0
set_seed(seed)


# [[[ CONFIG ]]]
# [USER]
timestamp = "2022_1230_1009_49"
epoch     = 113
tag       = "v2"

fl_chkpt = f"{timestamp}.epoch={epoch}.chkpt"

# Define the test set
# [USER]
size_sample_query = 1000
num_max_support   = 20
frac_support      = 0.4
size_batch        = 100
trans             = None
alpha             = 0.02
scan_rng          = np.linspace(-2, 2, 100)

# Initialize a log file...
init_logger(log_name = 'validate.query.test', timestamp = timestamp, returns_timestamp = False, saves_log = False)


# [[[ CHOOSE PDB ]]]
# Find the distribution of pdb over size...
path_dat = 'skopi/pdb_num.dat'
data = open(path_dat).readlines()
pdb_size_dict = {}
for d in data:
    pdb, mol_size = d.strip().split()
    pdb_size_dict[pdb] = int(mol_size)
size_list = list(pdb_size_dict.values())
size_nplist = np.array(size_list)
hy, hx = np.histogram(size_nplist, bins = 50)
size_pdb_dict = {}
for enum_i, (size_min, size_max) in enumerate(zip(hx[:-1], hx[1:])):
    size_pdb_dict[enum_i] = [ pdb for pdb, mol_size in pdb_size_dict.items() if size_min < mol_size < size_max ]

# Set up the right device for the computation...
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Choose pdbs from one bin...
bucket = 0
pdb_candidate_list = size_pdb_dict[bucket]
## num_pdb_for_test = 1
num_pdb_for_test = 100
num_pdb_for_test = min(len(pdb_candidate_list), num_pdb_for_test)
pdb_list = random.sample(pdb_candidate_list, k = num_pdb_for_test)
pdb_photon_res_dict = {}
for enum_pdb, pdb in enumerate(pdb_list):
    pdb_photon_res_dict[pdb] = []

    # [[[ DATASET ]]]
    # Set up parameters for an experiment...
    drc_dataset  = 'fastdata.h5/'
    fl_dataset   = f'{pdb}.relabel.pickle'
    path_dataset = os.path.join(drc_dataset, fl_dataset)

    # Load raw data...
    with open(path_dataset, 'rb') as fh:
        dataset_list = pickle.load(fh)

    # Scan through a list of shot to shot fluc setting...
    # [USER INPUT]
    for enum_photon, photon_exponent in enumerate(scan_rng):
        photon_scale = 10 ** photon_exponent

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

        for hit, idx_in_data_support in support_hit_to_idx_dict.items():
            if len(support_hit_to_idx_dict[hit]) > num_max_support:
                support_hit_to_idx_dict[hit] = random.sample(support_hit_to_idx_dict[hit], k = num_max_support)

        # Form query dataset...
        dataset_query = SPIOnlineDataset( dataset_list   = data_query, 
                                          size_sample    = size_sample_query,
                                          joins_metadata = False,
                                          trans          = trans, )

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
        diff_query_support_hit_to_idx_dict = {}
        for hit in hit_list:
            diff_query_support_hit_to_idx_dict[hit] = query_batch_emb[:,None] - support_batch_emb_dict[hit]

        dist_dict = {}
        for hit in hit_list:
            dist_dict[hit] = torch.sum(diff_query_support_hit_to_idx_dict[hit] * diff_query_support_hit_to_idx_dict[hit], dim = -1)

        # Use Torch tensor as a facilitate to get the predicted hit...
        enum_to_hit_dict = {}
        for enum_hit, hit in enumerate(hit_list):
            enum_to_hit_dict[enum_hit] = hit

            # Fetch the values and indices the support with the min dist when measured against the query...
            min_support_val, min_support_idx = dist_dict[hit].min(dim = -1)
            if enum_hit == 0:
                min_support_tensor = torch.zeros((len(hit_list), *min_support_val.shape))

            min_support_tensor[enum_hit] = min_support_val

        # Obtain the predicted hit...
        pred_hit_as_enum_list = min_support_tensor.min(dim = 0)[1]
        pred_hit_list = [ enum_to_hit_dict[enum.item()] for enum in pred_hit_as_enum_list ]

        # Obtain the real hit...
        real_hit_list = [ dataset_query[idx][1][1] for idx in query_idx_list ]

        # New container to store validation result (thus res_dict) for each label...
        res_dict = {}
        for hit in hit_list: res_dict[hit] = { i : [] for i in hit_list }

        for pred_hit, real_hit in zip(pred_hit_list, real_hit_list):
            res_dict[pred_hit][real_hit].append( None )

        pdb_photon_res_dict[pdb].append([photon_exponent, res_dict])

        acc = ConfusionMatrix(res_dict).get_metrics(1)[0]
        print(f"Working on {pdb}, {photon_exponent}, {acc}...")


    # [USER INPUT]
    fl_pdb_photon_res_dict = f'confusion_matrix.{timestamp}.epoch_{epoch}.{tag}.pickle'
    with open(fl_pdb_photon_res_dict, 'wb') as handle:
        pickle.dump(pdb_photon_res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
