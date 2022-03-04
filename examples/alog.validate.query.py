#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os
from deepprojection.utils import read_log
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

# File to analyze...
timestamp = "20220225234851"

# Validate mode...
istrain = True
mode_validate = 'train' if istrain else 'test'


# Locate the path of the log file...
fl_log   = f'{timestamp}.validate.query.{mode_validate}.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Fetch all labels...
record = log_dict["data"][0]
labels = [ item[:item.strip().find(":")][-1] for item in record[1:] ]

# New container to store validation result (thus res_dict) for each label...
res_dict = {}
## for label in labels: res_dict[label] = { True : [], False : [] }
for label in labels: res_dict[label] = { i : [] for i in labels }

# Process each record...
for i, record in enumerate(log_dict[ "data" ]):
    # Unpack each record...
    title_query, records_test = record[0].strip(), record[1:]

    # Retrieve title and loss for each test img...
    titles_test = []
    loss_test   = []
    for record_test in records_test:
        title_test, loss = record_test.split(":")

        titles_test.append(title_test.strip())
        loss_test.append(loss)

    # Find the most similar img measured by distance...
    dist_min           = min(loss_test)
    idx_mindist        = loss_test.index(dist_min)
    title_mindist = titles_test[idx_mindist]

    # Get the supposed result (match or not)...
    label_real = title_query[-1]
    label_pred = title_mindist[-1]
    ## is_accurate = label_real == label_pred

    res_dict[label_pred][label_real].append( (title_query, title_mindist) )

    ## # Tally the label...
    ## res_dict[label_pred][is_accurate].append( (title_query, title_mindist) )

# Formating purpose...
disp_dict = { "0" : "not sample",
              "1" : "single hit",
              "2" : "multi hit ",
              "9" : "background",  }

# Report multiway classification...
msgs = []
for label_pred in labels:
    disp_text = disp_dict[label_pred]
    msg = f"{disp_text}  |"
    for label_real in labels:
        num = len(res_dict[label_pred][label_real])
        msg += f"{num:>12d}"
    msgs.append(msg)

msg_header = " " * (msgs[0].find("|") + 1)
for label in labels: 
    disp_text = disp_dict[label]
    msg_header += f"{disp_text:>12s}"
print(msg_header)

msg_headerbar = "-" * len(msgs[0])
print(msg_headerbar)
for msg in msgs:
    print(msg)
