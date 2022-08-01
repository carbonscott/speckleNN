#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from pprint import pprint

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
        fp = sum( [ len(res_dict[label][i]) for i in labels_rest ] )
        tn = sum( sum( len(res_dict[i][j]) for j in labels_rest ) for i in labels_rest )
        fn = sum( [ len(res_dict[i][label]) for i in labels_rest ] )

        return tp, fp, tn, fn


    def get_metrics(self, label):
        # Early return if non-exist label is passed in...
        confusion = self.reduce_confusion(label)
        if confusion is None: return None

        # Calculate metrics...
        tp, fp, tn, fn = confusion
        if tp * tn * fp * fn: 
            accuracy    = (tp + tn) / (tp + tn + fp + fn)
            precision   = tp / (tp + fp)
            recall      = tp / (tp + fn)
            specificity = tn / (tn + fp)
            f1_inv      = (1 / precision + 1 / recall)
            f1          = 2 / f1_inv
        else:
            accuracy    = None
            precision   = None
            recall      = None
            specificity = None
            f1_inv      = None
            f1          = None


        return accuracy, precision, recall, specificity, f1


timestamp = '2022_0603_2226_44'

autolabel_dict = {}
drc_label      = 'labels'
fl_autolabel   = f'{timestamp}.auto.label.csv'
path_autolabel = os.path.join(drc_label, fl_autolabel)
with open(path_autolabel, 'r') as fh:
    lines = csv.reader(fh)
    next(lines)
    for line in lines:
        exp, run, event, label = line
        basename = (exp, int(run), int(event))
        autolabel_dict[basename] = label

label_dict = {}
exp = 'amo06516'
fl_label = f'{exp}.label.csv'
path_label = os.path.join(drc_label, fl_label)
with open(path_label, 'r') as fh:
    lines = csv.reader(fh)
    next(lines)
    for line in lines:
        exp, run, event, label = line
        basename = (exp, int(run), int(event))
        label_dict[basename] = label

res_dict = {}
labels = tuple(sorted((set(autolabel_dict.values()))))
for label in labels: res_dict[label] = { i : [] for i in labels }
for basename, label in label_dict.items():
    if not basename in autolabel_dict: continue
    if not label    in labels        : continue

    # Get the auto label...
    autolabel = autolabel_dict[basename]

    # Sort items according to auto label and their original label...
    res_dict[autolabel][label].append(basename)

# Get macro metrics...
macro_metric = MacroMetric(res_dict)

# Formating purpose...
disp_dict = { "0" : "not sample",
              "1" : "single hit",
              "2" : " multi hit",
              "9" : "background",
            }

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
        msg += f"{metric:>12.2f}" if isinstance(metric, float) else " " * 12
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
