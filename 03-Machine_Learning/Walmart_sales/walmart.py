#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




# =============================================================================
# Data loading and initial preprocessing
# =============================================================================

## Load data
df = pd.read_csv('./Walmart_Store_sales.csv')

## Remove records with NaNs in target 'Weekly_Sales'
df = df.dropna(subset=['Weekly_Sales'])

## convert 'Date' column to timestamp
df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')

## Add some date proxies to explore time periodicity
# Extracting the week day from the date makes it obvious that the data was
# aggregated on a weekly basis
df['Date'].dt.day_name()
# We can therefore explore annual periodicity by the week number (finest
# graining available) and by month for a more coarse graining yet more robust
# dependent variable
df['Week'] = df['Date'].dt.isocalendar().week
df['Month'] = df['Date'].dt.month


# =============================================================================
# Preliminary EDA
# =============================================================================

## Let us first validate our intuition about the sales dependence to the
## various features

# Sales are higher during holidays [box plot]
### [box plot holiday vs no holiday]

# Sales are higher near the end of the year
# fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
# ax[0].scatter(df['Month'], df['Weekly_Sales'])
# ax[1].scatter(df['Week'], df['Weekly_Sales'])

# Plot consumer price index (CPI) vs time for each store
for store_data in df.groupby('Store'):
    pass



