#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from deepprojection.utils import read_log

class MacroMetric:
    def __init__(self, res_dict):
        self.res_dict = res_dict


    def reduce_confusion(self, label):
        ''' Given a label, reduce multiclass confusion matrix to binary
            confusion matrix.
        '''
        res_dict    = self.res_dict
        labels      = res_dict.keys()
        labels_rest = [ i for i in labels if not i == label ]

        # Early return if non-exist label is passed in...
        if not label in labels: 
            print(f"label {label} doesn't exist!!!")
            return None

        # Obtain true positive...
        tp = len(res_dict[label][label])
        fp = sum( [ len(res_dict[i][label]) for i in labels_rest ] )
        tn = sum( sum( len(res_dict[i][j]) for j in labels_rest ) for i in labels_rest )
        fn = sum( [ len(res_dict[label][i]) for i in labels_rest ] )

        return tp, fp, tn, fn


    def get_metrics(self, label):
        # Early return if non-exist label is passed in...
        confusion = self.reduce_confusion(label)
        if confusion is None: return None

        # Calculate metrics...
        tp, fp, tn, fn = confusion
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        precision   = tp / (tp + fp)
        recall      = tp / (tp + fn)
        specificity = tn / (tn + fp) if tn + fp > 0 else None
        f1_inv      = (1 / precision + 1 / recall)
        f1          = 2 / f1_inv

        return accuracy, precision, recall, specificity, f1




# File to analyze...
timestamp = "20220316134804"

# Validate mode...
istrain = False
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

    res_dict[label_pred][label_real].append( (title_query, title_mindist) )

# Get macro metrics...
macro_metric = MacroMetric(res_dict)

# Formating purpose...
disp_dict = { "0" : "not sample",
              "1" : "single hit",
              "2" : " multi hit",
              "9" : "background",  }

# Report multiway classification...
msgs = []
for label_pred in labels:
    disp_text = disp_dict[label_pred]
    msg = f"{disp_text}  |"
    for label_real in labels:
        num = len(res_dict[label_pred][label_real])
        msg += f"{num:>12d}"

    metrics = macro_metric.get_metrics(label_pred)
    for metric in metrics:
        msg += f"{metric:>12.2f}"
    msgs.append(msg)

msg_header = " " * (msgs[0].find("|") + 1)
for label in labels: 
    disp_text = disp_dict[label]
    msg_header += f"{disp_text:>12s}"

for header in [ "accuracy", "precision", "recall", "specificity", "f1" ]:
    msg_header += f"{header:>12s}"
print(msg_header)

msg_headerbar = "-" * len(msgs[0])
print(msg_headerbar)
for msg in msgs:
    print(msg)
