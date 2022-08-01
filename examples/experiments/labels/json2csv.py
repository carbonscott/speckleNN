#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import csv

exp, run = "amo06516", 90
basename = f"{exp}_{run:04d}"
fl_json  = f"{basename}.label.json"

with open(fl_json, "r") as fh: 
    label_dict = json.load(fh)

# Sort dictionary according to the key (numerically)...
label_dict = {k: v for k, v in sorted(label_dict.items(), key=lambda item: int(item[0]))}

# Export the dictionary to csv...
fl_csv = f"{basename}.csv"

with open(fl_csv, 'w') as fh:
    fieldnames = ["exp", "run", "event_num", "label"]

    csv_writer = csv.DictWriter(fh, fieldnames = fieldnames)
    csv_writer.writeheader()

    for k, v in label_dict.items():
        csv_writer.writerow({ "exp" : exp, "run" : run, "event_num" : k, "label" : v })
