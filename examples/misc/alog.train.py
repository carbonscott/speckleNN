#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from deepprojection.utils import read_log

logging.basicConfig( format="%(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Locate the path of the log file...
## id_log   = '20220203115233'
id_log   = '20220203150247'
fl_log   = f'{id_log}.train.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Understand dataset...
uniq_dict = {}
comb_dict = {}
for anchor, pos, neg in log_dict[ "data" ]:
    anchor   = tuple(anchor.split())
    pos      = tuple(pos.split())
    neg      = tuple(neg.split())

    for i in (anchor, pos, neg):
        if not i in uniq_dict: uniq_dict[i] = True

    comb_str = f"{anchor[-1]}{pos[-1]}{neg[-1]}"
    if not comb_str in comb_dict: comb_dict[comb_str]  = 1
    else                        : comb_dict[comb_str] += 1

# Original datasets
data_dict = {}
for k in uniq_dict.keys():
    label = k[-1]
    if not label in data_dict: data_dict[label]  = 1
    else                     : data_dict[label] += 1
logger.info("___/ Tally Labels \___")
for k, v in data_dict.items(): logger.info(f"{k} : {v}")


# Tally combinations...
logger.info("___/ Tally Combination \___")
for k, v in comb_dict.items(): logger.info(f"{k} : {v}")
