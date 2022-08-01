#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from itertools import permutations
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
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        precision   = tp / (tp + fp)
        recall      = tp / (tp + fn)
        specificity = tn / (tn + fp) if tn + fp > 0 else None
        f1_inv      = (1 / precision + 1 / recall)
        f1          = 2 / f1_inv

        return accuracy, precision, recall, specificity, f1




# File to analyze...
## timestamp = "2022_0719_2156_29"
## timestamp = "2022_0726_1113_53"
timestamp = "2022_0726_1113_31"

# Locate the path of the log file...
## fl_log   = f'{timestamp}.validate.query.test'
fl_log   = f'{timestamp}.validate.query.test'
fl_log   = f'{fl_log}.log'
drc_log  = 'logs'
path_log = os.path.join(drc_log, fl_log)

# Read the file...
log_dict = read_log(path_log)

# Fetch all labels...
record = log_dict["data"][0]
labels = [ item[:item.strip().find(":")].split()[-1] for item in record[1:] ]

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
    label_real = title_query.split()[-1]
    label_pred = title_mindist.split()[-1]

    res_dict[label_pred][label_real].append( (title_query, title_mindist) )


false_label = '2'
false_labeled_item_dict = { k : v for k, v in res_dict[false_label].items() if k != false_label }

## false_label = '2'
## false_labeled_item_dict = { k : v for k, v in res_dict[false_label].items() if k != false_label }


# Export to csv file...
drc    =  'false_items'
fl_csv = f'{timestamp}.{false_label}.false.csv'
path_csv = os.path.join(drc, fl_csv)
with open(path_csv, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['query', 'support'])

    for k, v in false_labeled_item_dict.items():
        for false_item in v:
            csv_writer.writerow(false_item)
