#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import csv
from deepprojection.utils import read_log

# Choose a model based on timestamp...
timestamp    = "2022_0518_1827_35"

# Load embedding for the precomputed and the unlabeled...
drc_emb          = "embeds"
fl_precomp       = f"{timestamp}.precomp_emb.pt"
fl_comp          = f"{timestamp}.comp_emb.pt"
path_precomp     = os.path.join(drc_emb, fl_precomp)
path_comp        = os.path.join(drc_emb, fl_comp)
emb_precomp_dict = torch.load(path_precomp)
emb_comp         = torch.load(path_comp)

# Load metadata per case from log...
drc_log  = "logs"
fl_log   = f"{timestamp}.comp_emb.log"
path_log = os.path.join(drc_log, fl_log)
log      = read_log(path_log)

# Extract emb from dictionary...
for i, (label, emb_container) in enumerate(emb_precomp_dict.items()):
    emb = emb_container[0]

    if i == 0:
        num_labels = len(emb_precomp_dict)

        len_emb = len(emb)
        emb_precomp = torch.zeros(num_labels, len_emb)

        rng_start = 0
        rng_stride = 1

    emb_precomp[rng_start : rng_start + rng_stride] = emb
    rng_start += rng_stride

# Calculate squared distance between each unlabeled emb and each precomputed emb...
emb_diff_per_label = emb_comp[:, None] - emb_precomp[None, :]
emb_sqdist_per_label = torch.sum(emb_diff_per_label * emb_diff_per_label, dim = -1)    # Sum over emb rep dimension

# Find the label but represented in indexing...
label_encoded = torch.argmin(emb_sqdist_per_label, dim = -1)

# Save labels on a per exp per run basis...
idx_to_label = list(emb_precomp_dict.keys())
label_list = []
entry_list = log['data']
for i, (entry,) in enumerate(entry_list):
    _, _, exp, run, event_num, _ = entry.split()

    label_idx = label_encoded[i]
    label     = idx_to_label[label_idx]

    label_list.append((exp, run, event_num, label))

# Export it to csv...
drc_label = "labels"
fl_csv = f"{timestamp}.auto.label.csv"
path_csv = os.path.join(drc_label, fl_csv)
with open(path_csv, 'w') as fh:
    fieldnames = [ "exp", "run", "event_num", "label" ]
    csv_writer = csv.writer(fh)
    csv_writer.writerow(fieldnames)

    for each_label in label_list:
        csv_writer.writerow(each_label)
