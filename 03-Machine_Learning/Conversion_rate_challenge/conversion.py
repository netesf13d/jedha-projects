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
# fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
fig1 = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(
    nrows=1, ncols=3,
    width_ratios=[1, 1, 0.1], wspace=0.1,
    figure=fig1,
    left=0.1, bottom=0.2, right=0.95, top=0.77)
axs1 = gs.subplots(sharex=False, sharey=False, squeeze=True)

axs1[0].set_aspect('equal')
axs1[0].pcolor(new_pconv, cmap='plasma', edgecolors='k', lw=0.5)
axs1[0].set_xticks([0.5, 1.5, 2.5, 3.5], countries)
axs1[0].set_yticks([0.5, 1.5, 2.5], sources)
axs1[0].set_title("New visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    axs1[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


axs1[1].set_aspect('equal')
axs1[1].pcolormesh(rec_pconv, cmap='plasma', edgecolors='k', lw=0.5)
axs1[1].tick_params(top=False, labeltop=False,
                  bottom=False, labelbottom=False,
                  left=False, labelleft=False,
                  right=False, labelright=False)
# axs1[1].set_xlim(0, 1.2e5)
axs1[1].set_title("Recurring visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    axs1[0].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')
for (i, j), p in np.ndenumerate(rec_pconv):
    axs1[1].text(j+0.5, i+0.5, f'{100*p:.2f}', ha='center', va='center')


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



#%%

### Age data
## prepare age data
ages = np.arange(80)
age_data = {
    'counts_rec': np.zeros(80, dtype=int),
    'counts_new': np.zeros(80, dtype=int),
    'pconv_rec': np.zeros(80, dtype=float),
    'std_pconv_rec': np.zeros(80, dtype=float),
    'pconv_new': np.zeros(80, dtype=float),
    'std_pconv_new': np.zeros(80, dtype=float),
}
# new users
inew = df['new_user'] == 1
for i, c in df.loc[inew, 'age'].value_counts().items():
    age_data['counts_new'][i] = c
for i, p in df.loc[inew, ['age', 'converted']].groupby('age').mean()['converted'].items():
    age_data['pconv_new'][i] = p
    age_data['std_pconv_new'][i] = p*(1-p) / np.sqrt(age_data['counts_new'][i])
# recurrent users
irec = df['new_user'] == 0
for i, c in df.loc[irec, 'age'].value_counts().items():
    age_data['counts_rec'][i] = c
for i, p in df.loc[irec, ['age', 'converted']].groupby('age').mean()['converted'].items():
    age_data['pconv_rec'][i] = p
    age_data['std_pconv_rec'][i] = p*(1-p) / np.sqrt(age_data['counts_rec'][i])

## let us make a nice exponential fit
from scipy.optimize import curve_fit

def decay_exp(x: np.ndarray,
              A: float,
              tau: float)-> np.ndarray:
    """
    Exponential decay: f(x) = A * exp(- x / tau)
    """
    return A * np.exp(- x / tau)


x = ages[17:]
# new visitors
y_new = age_data['pconv_new'][17:]
yerr_new = age_data['std_pconv_new'][17:]
popt_new, pcov_new = curve_fit(decay_exp, x, y_new, p0=(0.05, 10),
                               sigma=np.where(yerr_new==0., 1., yerr_new))
# recurring visitors
y_rec = age_data['pconv_rec'][17:]
yerr_rec = age_data['std_pconv_rec'][17:]
popt_rec, pcov_rec = curve_fit(decay_exp, x, y_rec, p0=(0.05, 10),
                               sigma=np.where(yerr_rec==0., 1., yerr_rec))

## plot
fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

axs2[0].plot(ages, age_data['counts_new'], color='tab:blue',
             linewidth=1, label='new visitors')
axs2[0].plot(ages, age_data['counts_rec'], color='tab:orange',
             linewidth=1, label='recurrent visitors')
axs2[0].axvline(17, color='k', linewidth=1, linestyle='--')
axs2[0].text(11, 8500, '17 yo\ncutoff', ha='center', va='center')
axs2[0].set_xlim(0, 80)
axs2[0].set_ylim(-200, 10000)
axs2[0].set_title("Ages distribution")
axs2[0].set_xlabel('Age (years)')
axs2[0].set_ylabel('Counts')
axs2[0].legend()

axs2[1].errorbar(ages, age_data['pconv_new'], yerr=age_data['std_pconv_new'],
                 color='tab:blue', fmt='o', markersize=3, markeredgewidth=0.3,
                 label='new visitors')
axs2[1].plot(ages, decay_exp(ages, *popt_new),
             linewidth=1, color='tab:blue', label='exp decay fit')
axs2[1].errorbar(ages, age_data['pconv_rec'], yerr=age_data['std_pconv_rec'],
                 color='tab:orange', fmt='o', markersize=3, markeredgewidth=0.3,
                 label='recurrent visitors')
axs2[1].plot(ages, decay_exp(ages, *popt_rec),
             linewidth=1, color='tab:orange', label='exp decay fit')
axs2[1].axvline(17, color='k', linewidth=1, linestyle='--')

new_popt_txt = (f'A = {popt_new[0]:.3} +- {pcov_new[0, 0]:.2}\n'
                f'tau = {popt_new[1]:.3} +- {pcov_new[1, 1]:.2}\n')
axs2[1].text(14, 0.05, new_popt_txt, ha='center', va='center', fontsize=7)

rec_popt_txt = (f'A = {popt_rec[0]:.3} +- {pcov_rec[0, 0]:.2}\n'
                f'tau = {popt_rec[1]:.3} +- {pcov_rec[1, 1]:.2}\n')
axs2[1].text(35, 0.12, rec_popt_txt, ha='center', va='center', fontsize=7)

axs2[1].set_xlim(0, 80)
axs2[1].set_ylim(-0.005, 0.20)
axs2[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], [0, 5, 10, 15, 20])
axs2[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], minor=True)
axs2[1].set_title("Conversion probability distribution")
axs2[1].set_xlabel('Age (years)')
axs2[1].set_ylabel('Conversion probability (%)')
axs2[1].legend()

plt.show()

"""
Here we show the distribution of ages (left panel) and the conversion probability (right panel)
aggregated by `'source'` and `'country'`, but distinguishing new and recurrent visitors.
- The distribution of ages is the same for the two categories: a gaussian with a cutoff at 17 years old,
  centered at around 28 years old. The probability of having a new visitor is roughly twice that of a recurrent visitor.
- The newsletter subscription probability decays with age. The observed decay fits well with an exponential
  from which we recover very similar decay constants of about 14 years. The multiplicative factor is 5-6 times higher
  for recurrent visitors than for new visitors, and so is the subscription probability both both categories, *independently of age*.

These patterns are extremely regular and again reveal the artificial nature of the data. They would certainly not be occur in a realistic context.
"""


#%% 

### Number of pages visited
"""
We proceed in a similar fashion this time with the quantity `'total_pages_visited'`
"""
npages = np.arange(30)
page_data = {
    'counts_rec': np.zeros(30, dtype=int),
    'counts_new': np.zeros(30, dtype=int),
    'pconv_rec': np.zeros(30, dtype=float),
    'std_pconv_rec': np.zeros(30, dtype=float),
    'pconv_new': np.zeros(30, dtype=float),
    'std_pconv_new': np.zeros(30, dtype=float),
}

