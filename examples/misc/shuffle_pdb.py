#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import random
import numpy as np

# Set seed
seed = 0
random.seed(seed)
np.random.seed(seed)


# Read raw csv...
base_csv = "simulated.square_detector.datasets"
fl_csv   = f"{base_csv}.csv"
df       = pd.read_csv(fl_csv, header = 'infer')

# Find all uniq pdb names...
df_basename = df['basename'].str[:4]
pdbs = sorted(list(set(df_basename)))
num_pdbs = len(pdbs)

# Draw samples...
rate         = 80
frac         = rate / 100
num_sampled  = int(num_pdbs * frac)
pdbs_sampled = sorted(random.sample(pdbs, num_sampled))

# Save sampled pdb entries in csv...
select_sampled = df_basename.isin(pdbs_sampled)
df_sampled = df.loc[select_sampled]

# Save not sampled pdb entries in csv...
df_not_sampled = df.loc[~select_sampled]

# Save the sampled...
fl_sampled = f"{base_csv}.pdb_sampled.{rate}.csv"
df_sampled.to_csv(fl_sampled, index = False)

# Save the not sampled...
fl_not_sampled = f"{base_csv}.pdb_not_sampled.{rate}.csv"
df_not_sampled.to_csv(fl_not_sampled, index = False)
