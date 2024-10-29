#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import Colormap, LightSource
# from matplotlib.collections import PolyCollection



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
fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
axs1_cb = np.empty_like(axs1, dtype=object)

axs1[0].set_aspect('equal')
heatmap1 = axs1[0].pcolormesh(100*new_pconv, cmap='plasma',
                              edgecolors='k', lw=0.5)
axs1[0].set_xticks([0.5, 1.5, 2.5, 3.5], countries, rotation=60)
axs1[0].set_yticks([0.5, 1.5, 2.5], sources)
axs1[0].set_title("New visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    color = 'k' if p > np.max(new_pconv) / 3 else 'w'
    axs1[0].text(j+0.5, i+0.5, f'{100*p:.2f}', color=color,
                 ha='center', va='center')

divider1 = make_axes_locatable(axs1[0])
axs1_cb[0] = divider1.append_axes("right", size="7%", pad="5%")
fig1.add_axes(axs1_cb[0])
fig1.colorbar(heatmap1, cax=axs1_cb[0], orientation="vertical",
              ticklocation="right")
axs1_cb[0].set_yticks([0, 0.5, 1, 1.5, 2, 2.5])


axs1[1].set_aspect('equal')
heatmap2 = axs1[1].pcolormesh(100*rec_pconv, cmap='plasma',
                              edgecolors='k', lw=0.5)
axs1[1].set_xticks([0.5, 1.5, 2.5, 3.5], countries, rotation=60)
axs1[1].tick_params(left=False, labelleft=False)
axs1[1].set_title("Recurring visitors")
for (i, j), p in np.ndenumerate(rec_pconv):
    color = 'k' if p > np.max(rec_pconv) / 3 else 'w'
    axs1[1].text(j+0.5, i+0.5, f'{100*p:.2f}', color=color,
                 ha='center', va='center')

divider2 = make_axes_locatable(axs1[1])
axs1_cb[1] = divider2.append_axes("right", size="7%", pad="5%")
fig1.add_axes(axs1_cb[1])
fig1.colorbar(heatmap2, cax=axs1_cb[1], orientation="vertical",
              ticklocation="right")
axs1_cb[1].set_yticks([0, 5, 10, 15])


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

def prob_distrib(df: pd.DataFrame,
                 group_by: str,
                 variables: list[str],
                 xvals: list[np.ndarray],
                 )-> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    group_by : str
        DESCRIPTION.
    variables : list[str]
        DESCRIPTION.
    xvals : list[np.ndarray]
        DESCRIPTION.

    Returns
    -------
    group_vals : list
        DESCRIPTION.
    counts : np.ndarray
        DESCRIPTION.
    pconv : np.ndarray
        DESCRIPTION.
    std_pconv : TYPE
        DESCRIPTION.

    """
    group_vals = []
    counts = np.zeros((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                      dtype=float)
    pconv = np.full((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                    np.nan, dtype=float)

    for k, (group, gdf) in enumerate(df.groupby(group_by)):
        group_vals.append(group)
        for idx, c in gdf.loc[:, variables].value_counts().items():
            counts[k, *idx] = c
        pconv_df = gdf.loc[:, variables + ['converted']] \
                      .groupby(variables) \
                      .mean()['converted']
        for idx, p in pconv_df.items():
            idx = idx if isinstance(idx, tuple) else (idx,) # cast to tuple if no multiindex
            pconv[k, *idx] = p

    std_pconv = np.where(pconv * (1 - pconv) == 0., 1, pconv) / np.sqrt(counts)
    return group_vals, counts, pconv, std_pconv

# ## make data
# group_by = 'new_user'
# # variables = ['age', 'total_pages_visited']
# # xvals = [np.arange(80), np.arange(30)]
# variables = ['age']
# xvals = [np.arange(80)]

# group_vals = []
# counts = np.zeros((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
#                   dtype=float)
# pconv = np.full((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
#                 np.nan, dtype=float)

# for k, (group, gdf) in enumerate(df.groupby(group_by)):
#     group_vals.append(group)
#     for idx, c in gdf.loc[:, variables].value_counts().items():
#         counts[k, *idx] = c
#     pconv_df = gdf.loc[:, variables + ['converted']] \
#                   .groupby(variables) \
#                   .mean()['converted']
#     for idx, p in pconv_df.items():
#         idx = idx if isinstance(idx, tuple) else (idx,) # cast to tuple if no multiindex
#         pconv[k, *idx] = p

# std_pconv = np.where(pconv * (1 - pconv) == 0., 1, pconv) / np.sqrt(counts)



def distrib_plot(xvals: np.ndarray,
                 counts: np.ndaarray,
                 pconv: np.ndarray,
                 std_pconv: np.ndarray,
                 labels: list[str]):
    """
    

    Parameters
    ----------
    xvals : np.ndarray
        DESCRIPTION.
    counts : np.ndaarray
        DESCRIPTION.
    pconv : np.ndarray
        DESCRIPTION.
    std_pconv : np.ndarray
        DESCRIPTION.
    labels : list[str]
        DESCRIPTION.

    Returns
    -------
    fig : atplotlib Figure
        DESCRIPTION.
    axs : np.ndarray[matplotlib Axes]
        DESCRIPTION.

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    
    distribs = counts / np.sum(counts, axis=1, keepdims=True)
    for k, group in enumerate(group_vals):
        axs[0].plot(xvals, distribs[k], color=f'C{k}',
                    linewidth=1, label=labels[k])
    axs[0].axvline(17, color='k', linewidth=1, linestyle='--')
    axs[0].set_xlim(0, np.max(xvals)+1)
    axs[0].set_ylim(-np.max(distribs)*0.02, np.max(distribs)*1.02)
    axs[0].set_ylabel('Prob. density')

    for k, group in enumerate(group_vals):
        axs[1].errorbar(xvals, pconv[k], yerr=std_pconv[k],
                        color=f'C{k}', fmt='o', markersize=3, markeredgewidth=0.3,
                         label=labels[k])
    axs[1].set_xlim(0, np.max(xvals)+1)
    axs[1].set_ylim(-0.005, int(np.max(pconv[~np.isnan(pconv)])*120)/100)
    axs[1].set_title("Conversion probability distribution")
    axs[1].set_ylabel('Conversion probability')
    
    return fig, axs
    

# ## plot
# xvals = np.arange(80)
# pconv = pconv
# std_pconv = std_pconv
# labels = ['recur. visitors', 'new visitors']

# distribs = counts / np.sum(counts, axis=1, keepdims=True)

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# for k, group in enumerate(group_vals):
#     axs[0].plot(xvals, distribs[k], color=f'C{k}',
#                 linewidth=1, label=labels[k])
# axs[0].axvline(17, color='k', linewidth=1, linestyle='--')
# axs[0].set_xlim(0, np.max(xvals)+1)
# axs[0].set_ylim(-np.max(distribs)*0.02, np.max(distribs)*1.02)
# axs[0].set_ylabel('Prob. density')


# for k, group in enumerate(group_vals):
#     axs[1].errorbar(xvals, pconv[k], yerr=std_pconv[k],
#                     color=f'C{k}', fmt='o', markersize=3, markeredgewidth=0.3,
#                      label=labels[k])
# axs[1].set_xlim(0, np.max(xvals)+1)
# axs[1].set_ylim(-0.005, int(np.max(pconv[~np.isnan(pconv)])*120)/100)
# axs[1].set_title("Conversion probability distribution")
# axs[1].set_ylabel('Conversion probability')


xvals = np.arange(80)
group_vals, counts, pconv, std_pconv = prob_distrib(
    df, group_by='new_user', variables=['age'], xvals=[xvals])

fig, axs = distrib_plot(xvals, counts, pconv, std_pconv,
                        labels=['recur. visitors', 'new visitors'])
axs[0].legend()
axs[1].legend()
plt.show()



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
             linewidth=1, label='recur. visitors')
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
                 label='recur. visitors')
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
- The distribution of ages is the same for the two categories: a truncated gaussian with a cutoff at 17 years old,
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

# new users
inew = df['new_user'] == 1
for i, c in df.loc[inew, 'total_pages_visited'].value_counts().items():
    page_data['counts_new'][i] = c
for i, p in df.loc[inew, ['total_pages_visited', 'converted']].groupby('total_pages_visited').mean()['converted'].items():
    page_data['pconv_new'][i] = p
    page_data['std_pconv_new'][i] = p*(1-p) / np.sqrt(page_data['counts_new'][i])
# recurrent users
irec = df['new_user'] == 0
for i, c in df.loc[irec, 'total_pages_visited'].value_counts().items():
    page_data['counts_rec'][i] = c
for i, p in df.loc[irec, ['total_pages_visited', 'converted']].groupby('total_pages_visited').mean()['converted'].items():
    page_data['pconv_rec'][i] = p
    page_data['std_pconv_rec'][i] = p*(1-p) / np.sqrt(page_data['counts_rec'][i])


## Your beloved sigmoid fit
def sigmoid(x: np.ndarray,
            x0: float,
            a: float)-> np.ndarray:
    """
    Exponential decay: f(x) = 1 / (1 + exp(- a * (x - x0)))
    """
    return 1 / (1 + np.exp(- a * (x - x0)))


x = npages
# new visitors
y_new = page_data['pconv_new']
yerr_new = page_data['std_pconv_new']
popt_new, pcov_new = curve_fit(sigmoid, x, y_new, p0=(15, 1),
                               sigma=np.where(yerr_new==0., 1., yerr_new))
# recurring visitors
y_rec = page_data['pconv_rec']
yerr_rec = page_data['std_pconv_rec']
popt_rec, pcov_rec = curve_fit(sigmoid, x, y_rec, p0=(15, 1),
                               sigma=np.where(yerr_rec==0., 1., yerr_rec))


## plot
fig3, axs3 = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

axs3[0].plot(npages, page_data['counts_new'], color='tab:blue',
             linewidth=1, label='new visitors')
axs3[0].plot(npages, page_data['counts_rec'], color='tab:orange',
             linewidth=1, label='recur. visitors')
axs3[0].axvline(13, color='k', linewidth=1, linestyle='--')
# axs3[0].text(11, 8500, '17 yo\ncutoff', ha='center', va='center')
axs3[0].set_xlim(0, 30)
axs3[0].set_ylim(-200, 30000)
axs3[0].set_title("Ages distribution")
axs3[0].set_xlabel('# pages visited')
axs3[0].set_ylabel('Counts')
axs3[0].legend()

axs3[1].errorbar(npages, page_data['pconv_new'], yerr=page_data['std_pconv_new'],
                 color='tab:blue', fmt='o', markersize=3, markeredgewidth=0.3,
                 label='new visitors')
axs3[1].plot(npages, sigmoid(npages, *popt_new),
              linewidth=1, color='tab:blue', label='sigmoid fit')
axs3[1].errorbar(npages, page_data['pconv_rec'], yerr=page_data['std_pconv_rec'],
                 color='tab:orange', fmt='o', markersize=3, markeredgewidth=0.3,
                 label='recur. visitors')
axs3[1].plot(npages, sigmoid(npages, *popt_rec),
             linewidth=1, color='tab:orange', label='sigmoid fit')
axs3[1].axvline(13, color='k', linewidth=1, linestyle='--')

new_popt_txt = (f'x0 = {popt_new[0]:.3} +- {pcov_new[0, 0]:.2}\n'
                f'a = {popt_new[1]:.3} +- {pcov_new[1, 1]:.2}\n')
axs3[1].text(23, 0.8, new_popt_txt, ha='center', va='center', fontsize=7)

rec_popt_txt = (f'x0 = {popt_rec[0]:.3} +- {pcov_rec[0, 0]:.2}\n'
                f'a = {popt_rec[1]:.3} +- {pcov_rec[1, 1]:.2}\n')
axs3[1].text(7, 0.7, rec_popt_txt, ha='center', va='center', fontsize=7)

axs3[1].set_xlim(0, 30)
axs3[1].set_ylim(-0.005, 1.005)
# axs3[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], [0, 5, 10, 15, 20])
# axs3[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], minor=True)
axs3[1].set_title("Conversion probability distribution")
axs3[1].set_xlabel('# pages visited')
axs3[1].set_ylabel('Conversion probability')
axs3[1].legend()

plt.show()


"""
Here we show the distribution of number of pages visited (left panel) and the corresponding conversion probability (right panel)
built the same way as in the previous figure.
- Although we still have a factor ~2 between the total counts, we remark that website visits with more than 13 pages visited are
  mostly done by recurrent visitors. Accounting for the factor 2, this makes it more than twice likely that a > 13 pages visit
  is done by a recurrent visitor.
- The two distributions are thus not proportional. They actually look like
  [inverse-gamma distributions](https://en.wikipedia.org/wiki/Inverse-gamma_distribution).
- The newsletter subscription probability looks like a sigmoid for both new and recurring visitors. The inflection points are shifted,
  and looking at the 13 pages-visits threshold, we see that here the conversion probability is thrice for recurring users than for new users.
- Sigmoid fits with the slope and inflection point as parameters is also shown on the right panel. The slopes are roughly the same,
  the difference being the inflection point: 12.5 pages for recurring visitors vs 15 pages for new visitors.
- Finally, we note two outlier events on the right panel. ~30 pages visits with no subscription. We might consider removing these observations
  for training later on.

We again stress that such regularity is unlikely in any realistic context.
"""

#%%

### Effect of the number of pages visited for each country

"""
We noted above that the country of origin has a significant impact on the conversion probability.
We consider the two following hypotheses to explain this trend:
- The country of orgin has an impact on the distribution of number of pages visited, but not on the subscription threshold (the inflection point of the sigmoid).
  In this case, the country is expected to not be a good a predictor: the relevant information would be already contained in the number of pages visited.
- The pages visits distribution does not depend on the country, but the conversion threshold does. In this case, the country (more precisely, its correlation with the number of pages visted)
  would be a highly relevant predictor.
For comparison, the new/recurring character of visitors effect is a mixture of these two: it affects both the pages visits distribution and conversion threshold.
"""

countries = []
npages = np.arange(30)
page_counts = np.zeros((4, 30), dtype=float)
pconv = np.zeros((4, 30), dtype=float)
std_pconv = np.zeros((4, 30), dtype=float)

for k, (country, df_country) in enumerate(df.groupby('country')):
    countries.append(country)
    for i, c in df_country.loc[:, 'total_pages_visited'].value_counts().items():
        page_counts[k, i] = c
    for i, p in df_country.loc[:, ['total_pages_visited', 'converted']].groupby('total_pages_visited').mean()['converted'].items():
        pconv[k, i] = p
        std_pconv[k, i] = p*(1-p) / np.sqrt(page_counts[k, i])

page_distrib = page_counts / np.sum(page_counts, axis=1, keepdims=True)

## plot
fig4, axs4 = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

for k, country in enumerate(countries):
    axs4[0].plot(npages, page_distrib[k], color=f'C{k+1}',
                 linewidth=1, label=country)
axs4[0].axvline(13, color='k', linewidth=1, linestyle='--')
axs4[0].set_xlim(0, 30)
axs4[0].set_ylim(0, 0.16)
axs4[0].set_title("# pages visited distribution")
axs4[0].set_xlabel('# pages visited')
axs4[0].set_ylabel('Probability')
axs4[0].legend()

for k, country in enumerate(countries):
    axs4[1].errorbar(npages, pconv[k], yerr=std_pconv[k],
                     color=f'C{k+1}', fmt='o', markersize=3, markeredgewidth=0.3,
                     label=country)
axs4[1].axvline(13, color='k', linewidth=1, linestyle='--')

axs4[1].set_xlim(0, 30)
axs4[1].set_ylim(-0.005, 1.005)
axs4[1].set_title("Conversion probability distribution")
axs4[1].set_xlabel('# pages visited')
axs4[1].set_ylabel('Conversion probability')
axs4[1].legend()

plt.show()

"""
Here we show, for each country, the distribution of pages visited (left panel) and the conversion probability as a function of the number of pages visited (right panel).
- The behavior of users from `'Germany'`, `'UK'` and `'US'` is quite similar both in terms of page visits and conversion probability thresholding with a tendency Germany < UK < US
which matches the differences observed in figure 1.
- The behavior difference of users from `'China'` is striking. First, the distibution of pages visits is different, with much less visits of more than 13 pages, those which are associated to newsletter subscription.
Second, the threshold seems to be set at a significantly higher level than other countries. These two factors explain the much lower conversion rates observed in figure 1.
"""


#%% 

### Combined effect of age and number of pages visited
pconv_new = np.zeros((80, 30), dtype=float)
std_pconv_new = np.zeros((80, 30), dtype=float)
pconv_rec = np.zeros((80, 30), dtype=float)
std_pconv_rec = np.zeros((80, 30), dtype=float)

# it's getting quite complex uh ?
# new users
pconv_new_counts_df = df.loc[inew, ['age', 'total_pages_visited']].value_counts()
pconv_new_df = df.loc[inew, ['age', 'total_pages_visited', 'converted']] \
                 .groupby(['age', 'total_pages_visited']) \
                 .mean()['converted']
for (i, j), p in pconv_new_df.items():
    pconv_new[i, j] = p
for (i, j), c in pconv_new_counts_df.items():
    std_pconv_new[i, j] = pconv_new[i, j] / np.sqrt(c)
# recurrent visitors
pconv_rec_counts_df = df.loc[irec, ['age', 'total_pages_visited']].value_counts()
pconv_rec_df = df.loc[irec, ['age', 'total_pages_visited', 'converted']] \
                 .groupby(['age', 'total_pages_visited']) \
                 .mean()['converted']
for (i, j), p in pconv_rec_df.items():
    pconv_rec[i, j] = p
for (i, j), c in pconv_rec_counts_df.items():
    std_pconv_rec[i, j] = pconv_rec[i, j] / np.sqrt(c)


x, y = np.meshgrid(npages[:25], ages[17:46])

fig4, axs4 = plt.subplots(
    nrows=1, ncols=2,
    gridspec_kw={'left':0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1},
    subplot_kw=dict(projection='3d'))

# new visitors
axs4[0].view_init(elev=20, azim=-110)
surf = axs4[0].plot_surface(
    x, y, pconv_new[17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs4[0].set_xlim(0, 24)
axs4[0].set_ylim(17, 45)
axs4[0].tick_params(pad=0)
axs4[0].set_title('New visitors')
axs4[0].set_xlabel('# pages visited')
axs4[0].set_ylabel('age')
axs4[0].set_zlabel('conv. prob.')

# recurring visitors
axs4[1].view_init(elev=20, azim=-110)
surf = axs4[1].plot_surface(
    x, y, pconv_rec[17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs4[1].set_xlim(0, 24)
axs4[1].set_ylim(17, 45)
axs4[1].tick_params(pad=0)
axs4[1].set_title('Recurring visitors')
axs4[1].set_xlabel('# pages visited')
axs4[1].set_ylabel('age')
axs4[1].set_zlabel('conv. prob.')

plt.show()

"""
The figure shows the conversion probability as a function of age and number of pages visited
for both new (left panel) and recurring (right panel) visitors.
- the sigmoid dependence of the conversion probability vs. number of pages visited is valid over the whole age range.
- In both conditions, there is a small linear dependence of the inflection point with age, with a lower inflection point for younger visitors.
"""

#%%

from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                     StratifiedKFold,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score,
                             confusion_matrix,
                             roc_curve,
                             auc)


from typing import Callable

def evaluate_model(preprocessor: Callable,
                   predictor: Callable,
                   name_suffix: str = 'EXAMPLE')-> np.ndarray:
    """
    Evaluate the model on the test set. The predictions are exported as
    'conversion_data_test_predictions_{name_suffix}.csv'

    Parameters
    ----------
    preprocessor : Callable: pd.DataFrame -> T
        The feature preprocessor.
    prediction_engine : Callable: T -> np.ndarray
        The prediciton engine of the model.
    name_suffix : str, optional
        Suffix appended to the output filename. The default is 'EXAMPLE'.
    
    Returns
    -------
    Y_pred : 1D np.ndarray
        The model's predictions on the test set.
    """
    df = pd.read_csv('./conversion_data_test.csv')
    X = preprocessor(df)
    Y_pred = predictor(X)
    
    fname = f'conversion_data_test_predictions_{name_suffix}.csv'
    np.savetxt(fname, Y_pred, fmt='%d', header='converted', comments='')
    
    return Y_pred

#%% TODO basic linregress



#%%

categorical_vars = ['country', 'source']
quantitative_vars = ['age', 'new_user', 'total_pages_visited']


## Data to fit
y = df.loc[:, 'converted']
X_df = df.loc[:, [c for c in df.columns if c != 'converted']]

## One hot encoding
X_df = pd.get_dummies(X_df)
X = X_df.drop(['country_US', 'source_Seo'], axis=1)


## model evaluation
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', LogisticRegression())])
# scores = cross_validate(pipeline, X, Y, cv=5, scoring='f1')
model = TunedThresholdClassifierCV(pipeline, scoring='f1', cv=10,
                                   random_state=None,
                                   store_cv_results=True)
model.fit(X, y)
best_thr = model.best_threshold_
best_score = model.best_score_
print(f'best threshold = {best_thr:.8f}',
      f'best F1-score = {best_score:.8f}')

#%%
## metrics: confusion matrix and ROC
n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
clf = Pipeline([('scaler', StandardScaler()),
                      ('classifier', LogisticRegression())])

mean_cm = np.zeros((2, 2), dtype=float) # confusion matrix
mean_fpr = np.linspace(0, 1, 101) # false-positive rates
mean_tpr = np.zeros(101, dtype=float) # true-positive rates
thr_fpr, thr_tpr = 0, 0 # fpr, tpr at threshold
for i, (itr, ival) in enumerate(cv.split(X, y)):
    X_v, y_v = X.iloc[ival], y.iloc[ival]
    clf.fit(X.iloc[itr], y.iloc[itr])
    # confusion matrix
    y_pred = clf.predict_proba(X_v)[:, 1] > best_thr
    mean_cm += confusion_matrix(y_v, y_pred) / len(X)
    # ROC
    y_decision = clf.decision_function(X_v)
    fpr, tpr, thr = roc_curve(y_v, y_decision)
    print(thr)
    mean_tpr += np.interp(mean_fpr, fpr, tpr, left=0, right=1) / n_splits
    thr_fpr += np.interp(best_thr, thr, fpr) / n_splits
    thr_tpr += np.interp(best_thr, thr, tpr) / n_splits
mean_tpr[0] = 0
mean_auc = auc(mean_fpr, mean_tpr)



#%%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})')
ax.plot(thr_fpr, thr_tpr, marker='X', markersize=5)
ax.text(0.8, 0.93, f'threshold = {best_thr:.4f}\nF1-score = {best_score:.4f}',
        ha='center', va='center')
ax.legend(loc=4)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.grid(visible=True, alpha=0.3)
plt.show()



## preprocess
# fenc = StandardScaler()
# X_tr = fenc.fit_transform(X_tr)

# ## split
# X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y)


# ##### training pipeline #####
# ## preprocess
# featureencoder = StandardScaler()
# X_tr = featureencoder.fit_transform(X_tr)

# ## train
# classifier = LogisticRegression() # 
# classifier.fit(X_tr, Y_tr)

# ## train set predicitions
# Y_tr_pred = classifier.predict(X_tr)



# ##### performance assessment #####
# ## preprocess
# X_val = featureencoder.transform(X_val)

# ## make predicitions
# Y_val_pred = classifier.predict(X_val)


# ## metrics
# print(f'Train set F1-score: {f1_score(Y_tr, Y_tr_pred)}')
# print(f'Test set F1-score: {f1_score(Y_val, Y_val_pred)}')
# print("Train set confusion matrix\n",
#       confusion_matrix(Y_tr, Y_tr_pred, normalize='all'))
# print("Test set confusion matrix\n",
#       confusion_matrix(Y_val, Y_val_pred, normalize='all'))

#%%
##### testing pipeline #####
from hashlib import shake_256

def preprocessor_baseline(dataframe: pd.DataFrame)-> np.ndarray:
    df = pd.get_dummies(dataframe)
    df = df.drop(['country_US', 'source_Seo'], axis=1)
    return df

def predictor_baseline(X: np.ndarray)-> np.ndarray:
    return model.predict(X)

models = [
    b'baseline',
    b'linreg_threhold_adjust',
    ]
models = {m: shake_256(m).hexdigest(4) for m in models}


evaluate_model(preprocessor_baseline,
               predictor_baseline,
               name_suffix=models['linreg_threhold_adjust'])



