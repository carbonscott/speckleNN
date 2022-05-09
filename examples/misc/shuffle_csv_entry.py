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

# Draw samples...
rate = 80
df_sampled = df.sample(frac = rate / 100)

# Obtain those not sampled...
df_not_sampled = df[~df.index.isin(df_sampled.index)]

# Save the sampled...
fl_sampled = f"{base_csv}.sampled.{rate}.csv"
df_sampled.to_csv(fl_sampled, index = False)

# Save the not sampled...
fl_not_sampled = f"{base_csv}.not_sampled.{rate}.csv"
df_not_sampled.to_csv(fl_not_sampled, index = False)
