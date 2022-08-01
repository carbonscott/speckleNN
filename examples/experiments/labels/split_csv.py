#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv

exp, runs = "amo06516", [90, 91, 94, 96, 102]

# Read all labels from the master csv...
fl_csv = f"{exp}.label.csv"

run_dict = {}
with open(fl_csv, 'r') as fh:
    lines = csv.reader(fh)

    next(lines)

    for line in lines:
        exp, run, event_num, label = line

        basename = (exp, int(run))
        record   = int(event_num), label
        if not basename in run_dict: run_dict[basename] = [ record ]
        else                       : run_dict[basename].append(record)


filednames = [ "exp", "run", "event_num", "label" ]
for run in runs:
    fl_csv = f"{exp}_{run:04d}.label.csv"

    basename = (exp, run)
    lines = run_dict[basename]
    with open(fl_csv, 'w') as fh:
        csv_writer = csv.writer(fh)

        csv_writer.writerow(filednames)
        for line in lines:
            event_num, label = line
            record = exp, run, event_num, label
            csv_writer.writerow(record)
