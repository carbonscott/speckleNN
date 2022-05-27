#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import csv
import argparse

parser = argparse.ArgumentParser(
description = 
"""Create a nolabel file with hit events."""
,formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-e", "--exp",
    required = True,
    type     = str,
    help     = """Experiment name.""",)

parser.add_argument("-r", "--run",
    required = True,
    type     = int,
    help     = """Run number""",)

args = parser.parse_args()


exp, run = args.exp, args.run

path_cxi = f"{exp}_{run:04d}.cxi"
with h5py.File(path_cxi, 'r') as fh:
    hit_list = fh['LCLS/eventNumber'][()]

hit_dict = { i : "-1" for i in sorted(hit_list) }

path_csv = f"{exp}_{run:04d}.label.csv"
with open(path_csv, 'w') as fh:
    fieldnames = [ "event_num", "label" ]
    csv_writer = csv.DictWriter(fh, fieldnames = fieldnames)

    csv_writer.writeheader()
    for k, v in hit_dict.items():
        csv_writer.writerow( { "event_num" : k, "label" : v } )
