#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os
from deepprojection.utils import read_log

# File to analyze...
## id_log = '20220203115233'    # 0, 1, 2
id_log = '20220203150247'    # 1, 2

# Configure the location to run the job...
drc_cwd = os.getcwd()

# Set up the log file...
fl_alog = f"{id_log}.validate.alog"
DRCALOG = "analysis"
prefixpath_alog = os.path.join(drc_cwd, DRCALOG)
if not os.path.exists(prefixpath_alog): os.makedirs(prefixpath_alog)
path_alog = os.path.join(prefixpath_alog, fl_alog)

logging.basicConfig( filename = path_alog,
                     filemode = 'w',
                     format="%(message)s",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Locate the path of the log file...
fl_log   = f'{id_log}.validate.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Understand dataset...
uniq_dict = {}
comb_dict = {}
for anchor, pos, neg, loss in log_dict[ "data" ]:
    anchor   = tuple(anchor.split())
    pos      = tuple(pos.split())
    neg      = tuple(neg.split())
    loss     = np.float64(loss)

    for i in (anchor, pos, neg):
        if not i in uniq_dict: uniq_dict[i] = True

    comb_str = f"{anchor[-1]}{pos[-1]}{neg[-1]}"
    if not comb_str in comb_dict: 
        comb_dict[comb_str] = {}
        comb_dict[comb_str]["count"] = 1
        comb_dict[comb_str]["loss"]  = [ loss ]
    else: 
        comb_dict[comb_str]["count"] += 1
        comb_dict[comb_str]["loss"].append(loss)

# Original datasets
data_dict = {}
for k in uniq_dict.keys():
    label = k[-1]
    if not label in data_dict: data_dict[label]  = 1
    else                     : data_dict[label] += 1
logger.info("___/ Tally Labels \___")
for k, v in data_dict.items(): logger.info(f"label = {k} : count = {v}")


# Tally combinations...
logger.info("___/ Tally Combination \___")
for k, v in comb_dict.items(): 
    count     = v["count"]
    loss      = v["loss"]
    loss_mean = np.mean(loss)
    loss_std  = np.std(loss)
    logger.info(f"comb = {k} : count = {count}, mean(+-)std = {loss_mean:6.4f}(+-){loss_std:6.4f}")
