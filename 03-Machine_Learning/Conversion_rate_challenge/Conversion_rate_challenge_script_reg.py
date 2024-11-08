# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:49:22 2024

@author: netes
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit


# =============================================================================
# Data loading and initial preprocessing
# =============================================================================

# Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]

features = list(df.columns[:-1])
pconv = df.groupby(features).mean()

data = pd.DataFrame.from_dict(
    ({c: v.to_numpy() for c, v in pconv.index.to_frame().items()}
     | {'pconv': pconv['converted'].to_numpy()})
)