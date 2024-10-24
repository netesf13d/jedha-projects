#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# =============================================================================
# Data loading and initial preprocessing
# =============================================================================

## Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]


# test_df = pd.read_csv('./conversion_data_test.csv')
# test_df.describe(include='all')
# """
# The distribution of the different observations is essentially the same
# as that of the training set in terms of mean and standard deviation.
# """


#%%
# =============================================================================
# Preliminary EDA
# =============================================================================


## 
df.groupby(['country', 'converted']).count()['age'] # the name 'age' is irrelevant here
##
df.groupby(['source', 'converted']).count()['age']

df1 = df.groupby(['new_user', 'source', 'country']).mean()['converted']


countries = ['China', 'Germany', 'UK', 'US']
sources = ['Ads', 'Direct', 'Seo']


# new user
new_pconv = np.empty((len(sources), len(countries)), dtype=float)
for i, source in enumerate(sources):
    for j, country in enumerate(countries):
        new_pconv[i, j] = df1.loc[1, source, country]

# recurring user
rec_pconv = np.empty((len(sources), len(countries)), dtype=float)
for i, source in enumerate(sources):
    for j, country in enumerate(countries):
        rec_pconv[i, j] = df1.loc[0, source, country]


# TODO finish plot
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(
    nrows=1, ncols=3,
    width_ratios=[1, 1, 0.1], wspace=0.1,
    figure=fig,
    left=0.1, bottom=0.2, right=0.95, top=0.77)
ax = gs.subplots(sharex=False, sharey=False, squeeze=True)

ax[0].set_aspect('equal')
ax[0].pcolor(new_pconv, cmap='plasma', edgecolors='k', lw=0.5)
ax[0].set_xticks([0.5, 1.5, 2.5, 3.5], countries)
ax[0].set_yticks([0.5, 1.5, 2.5], sources)
ax[0].set_title("New visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    ax[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


ax[1].set_aspect('equal')
ax[1].pcolormesh(rec_pconv, cmap='plasma', edgecolors='k', lw=0.5)
ax[1].tick_params(top=False, labeltop=False,
                  bottom=False, labelbottom=False,
                  left=False, labelleft=False,
                  right=False, labelright=False)
# ax[1].set_xlim(0, 1.2e5)
ax[1].set_title("Recurring visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    ax[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')
for (i, j), p in np.ndenumerate(rec_pconv):
    ax[1].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


# cb = fig.colorbar(yzprof, cax=ax[2], orientation="horizontal",
#                   ticklocation="top", extend="max", extendfrac=0.03)

plt.show()

"""
Here we show heatmaps of the newsletter subscription probabilility for the
categorical features `'source'` and `'country'`, for both new user and recurring users.
We can make the following remarks:
- Recurring visitors are 2.5 - 8 times more likely to subscribe to the newsletter.
- Chinese visitors are much less likely to subscribe than others, and US visitors tend to subscribe less than europeans.
- The way by which visitors reached the website has little impact on their eventual subscription.
"""


##

pconv_vs_age = df.loc[:, ['age', 'converted']].groupby('age').mean()




# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))


# ax[0].set_aspect('equal')
# ax[0].pcolor(new_pconv, cmap='plasma', edgecolors='k', lw=0.5)
# ax[0].set_xticks([0.5, 1.5, 2.5, 3.5], countries)
# ax[0].set_yticks([0.5, 1.5, 2.5], sources)
# ax[0].set_title("New visitors")
# for (i, j), p in np.ndenumerate(new_pconv):
#     ax[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


# ax[1].set_aspect('equal')
# ax[1].pcolormesh(rec_pconv, cmap='plasma', edgecolors='k', lw=0.5)
# ax[1].tick_params(top=False, labeltop=False,
#                   bottom=False, labelbottom=False,
#                   left=False, labelleft=False,
#                   right=False, labelright=False)
# # ax[1].set_xlim(0, 1.2e5)
# ax[1].set_title("Recurring visitors")
# for (i, j), p in np.ndenumerate(new_pconv):
#     ax[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')
# for (i, j), p in np.ndenumerate(rec_pconv):
#     ax[1].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


# plt.show()


