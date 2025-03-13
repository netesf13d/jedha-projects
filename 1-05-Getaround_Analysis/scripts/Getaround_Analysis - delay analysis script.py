# -*- coding: utf-8 -*-
"""
Script for Getaround project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %% Loading
"""
## <a name="loading"></a> Data loading
"""

##
df = pd.read_excel('../data/get_around_delay_analysis.xlsx')
df

"""
The dataset is split in two parts, with 21310 observations each.
- The first dataframe, `pp_df`, contains the information that we have *before* a car is rent.
We will use these features as predictors for our machine learning model
- The second dataframe contains information relative to the *outcome* of a rental. This data cannot be used to make predictions.
However, we will use it to build target feature to train our ML model.

We should include all the data available in our exploratory data analysis. we build a combined dataframe `eda_df` for this purpose.
"""


"""
`'checkin_type'`: `{'mobile', 'connect'}` 
`'state'`: `{'ended', 'cancelled'}`, whether the rental was cancelled or not
`'delay_at_checkout_in_minutes'`: delay between the expected and actual checkout times (positive means checkout later than expected)
`'time_delta_with_previous_rental_in_minutes'`: How long the car stood by unrented (longer is a loss of money)
``
"""


# %% EDA
"""
## <a name="eda"></a> Preliminary data analysis


"""

# %%
"""
### some graphs


- follow the trajectory of a car
"""

##
df_ = df.groupby(['checkin_type', 'state']).count()['car_id']
df_

##
df_ / df_.T.groupby('checkin_type').sum()

"""
Higher probability of cancellation when connect
"""




# %%
"""
### Checkout delay


"""

##
df_ = df.groupby(['checkin_type', 'state']).agg(lambda x: x.isnull().sum())
df_

"""
Almost all null `delay_at_checkout_in_minutes` when canceled
"""


##
delay_vals = np.logspace([0], [5], 21)
checkout_distrib = {}
for (checkin,), df_ in df.groupby(['checkin_type']): #, 'state']):
    data = df_['delay_at_checkout_in_minutes'].to_numpy()
    data = data[~np.isnan(data)]
    checkout_distrib[checkin] = [
        np.sum(data >= delay_vals, axis=1) / len(data),
        np.sum(data <= -delay_vals, axis=1) / len(data),
        ]


##
fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4), dpi=200,
    gridspec_kw={'left': 0.075, 'right': 0.97, 'top': 0.85, 'bottom': 0.13,
                 'wspace': 0.18})
fig1.suptitle('Figure 1: checkout delay distribution', x=0.02, ha='left')

labels = [r'$P\,(\mathrm{delay} \geq \tau)$', r'$P\, (\mathrm{delay} \leq - \tau)$']

axs1[0].set_title("Mobile checkin")
line0, = axs1[0].plot(delay_vals, checkout_distrib['mobile'][0],
                      linestyle='', marker='.', markersize=10, color='tab:blue')
line1, = axs1[0].plot(delay_vals, checkout_distrib['mobile'][1],
             linestyle='', marker='.', markersize=10, color='tab:orange')
axs1[0].text(0.035, 0.06, r'$P\,(\mathrm{delay} \geq 0) = '
             fr'{checkout_distrib["mobile"][0][0]:.3f}$',
             transform=axs1[0].transAxes, fontsize=9,
             bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '0.92'})
axs1[0].grid(visible=True, linewidth=0.3)
axs1[0].set_xscale('log')
axs1[0].set_xlim(1, 1e5)
axs1[0].set_yscale('log')
axs1[0].set_ylim(1e-4, 1)
axs1[0].set_xlabel(r"Checkout delay $\tau$ (min)")
axs1[0].set_ylabel('Inverse cumm. prob.')
axs1[0].legend(handles=[line0, line1], labels=labels)


axs1[1].set_title("Getaround connect checkin")
line0, = axs1[1].plot(delay_vals, checkout_distrib['connect'][0],
                      linestyle='', marker='.', markersize=10, color='tab:blue')
line1, = axs1[1].plot(delay_vals, checkout_distrib['connect'][1],
             linestyle='', marker='.', markersize=10, color='tab:orange')
axs1[1].text(0.035, 0.06, r'$P\,(\mathrm{delay} \geq 0) = '
             fr'{checkout_distrib["connect"][0][0]:.3f}$',
             transform=axs1[1].transAxes, fontsize=9,
             bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '0.92'})
axs1[1].grid(visible=True, linewidth=0.3)
axs1[1].set_xscale('log')
axs1[1].set_xlim(1, 1e5)
axs1[1].set_yscale('log')
axs1[1].set_ylim(1e-4, 1)
axs1[1].set_xlabel(r"Checkout delay $\tau$ (min)")
axs1[1].legend(handles=[line0, line1], labels=labels)


plt.show()

"""
Figure 1 presents the distribution of checkout delays.
"""


# %%
"""
### Delay with previous rental


"""

##
rental_delay = {'mobile': [0, 0], 'connect': [0, 0]}
for (checkin, state), df_ in df.groupby(['checkin_type', 'state']):
    if state == 'ended':
        rental_delay[checkin][0] = df_['time_delta_with_previous_rental_in_minutes'].to_numpy()
    if state == 'canceled':
        rental_delay[checkin][1] = df_['time_delta_with_previous_rental_in_minutes'].to_numpy()

##
delay_bins = np.linspace(-30, 750, 14)
delay_vals = (delay_bins[1:] + delay_bins[:-1]) / 2

## 'mobile' delay histograms : ended, canceled, cancelation prob 
delay_hist_me, _ = np.histogram(rental_delay['mobile'][0], bins=delay_bins)
delay_hist_mc, _ = np.histogram(rental_delay['mobile'][1], bins=delay_bins)
delay_hist_mfrac = delay_hist_mc / (delay_hist_mc + delay_hist_me)

## 'connect' delay histograms : ended, canceled, cancelation prob 
delay_hist_ce, _ = np.histogram(rental_delay['connect'][0], bins=delay_bins)
delay_hist_cc, _ = np.histogram(rental_delay['connect'][1], bins=delay_bins)
delay_hist_cfrac = delay_hist_cc / (delay_hist_cc + delay_hist_ce)

##
fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4), dpi=200,
    gridspec_kw={'left': 0.07, 'right': 0.94, 'top': 0.85, 'bottom': 0.13,
                 'wspace': 0.24})
axs2_twin = [ax.twinx() for ax in axs2]
fig2.suptitle('Figure 2: Distribution of delays with previous rental', x=0.02, ha='left')



handles = [Patch(facecolor='tab:blue', alpha=1, label='ended'),
           Patch(facecolor='tab:orange', alpha=1, label='canceled')]
labels = ['ended', 'canceled', 'cancelation prob.']

axs2[0].set_title("Mobile checkin")
axs2[0].hist(rental_delay['mobile'], bins=np.linspace(-30, 750, 14),
             stacked=False, density=False)

line, = axs2_twin[0].plot(delay_vals, delay_hist_mfrac, color='tab:green',
                  marker='o', markersize=5)
axs2_twin[0].set_ylim(0, 0.425)
axs2_twin[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axs2_twin[0].set_yticks([0.05, 0.15, 0.25, 0.35], minor=True)

axs2[0].grid(visible=True, linewidth=0.3)
axs2[0].set_xlim(-30, 750)
axs2[0].set_xticks(np.linspace(0, 720, 7))
axs2[0].set_xticks(np.linspace(60, 660, 6), minor=True)
axs2[0].set_xlabel("Delay with previous rental (min)")
axs2[0].set_ylim(0, 170)
axs2[0].set_ylabel("Counts")
axs2[0].legend(handles=handles + [line], labels=labels)


axs2[1].set_title("Getaround connect checkin")
axs2[1].hist(rental_delay['connect'], bins=np.linspace(-30, 750, 14), # np.linspace(-15, 735, 26)
             stacked=False, density=False)

line, = axs2_twin[1].plot(delay_vals, delay_hist_cfrac, color='tab:green',
                          marker='o', markersize=5)
axs2_twin[1].set_ylim(0, 0.425)
axs2_twin[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axs2_twin[1].set_yticks([0.05, 0.15, 0.25, 0.35], minor=True)

axs2[1].grid(visible=True, linewidth=0.3)
axs2[1].set_xlim(-30, 750)
axs2[1].set_xticks(np.linspace(0, 720, 7))
axs2[1].set_xticks(np.linspace(60, 660, 6), minor=True)
axs2[1].set_ylim(0, 170)
axs2[1].set_xlabel("Delay with previous rental (min)")
axs2[1].legend(handles=handles + [line], labels=labels)


plt.show()

"""
Figure 2
"""

# %%

"""
### Cancel probability


"""

##
prev_rental_cols = ['rental_id', 'delay_at_checkout_in_minutes']
curr_rental_cols = ['previous_ended_rental_id', 'checkin_type', 'state',
                    'time_delta_with_previous_rental_in_minutes']
df_prev = df.loc[:, prev_rental_cols]
df_curr = df.loc[:, curr_rental_cols]
df2 = pd.merge(df_prev, df_curr, how='inner', left_on='rental_id',
               right_on='previous_ended_rental_id')
df2 = df2.assign(is_canceled=(df2['state'] == 'canceled'))

df2.describe(include='all')
# df_ = df_.loc[:, ['delay_at_checkout_in_minutes', 'checkin_type', 'state', 'time_delta_with_previous_rental_in_minutes']]

##
# delay_vals = np.logspace([0], [4], 21)
# checkout_distrib2 = {}
# for item, df_ in df2.groupby(['checkin_type', 'state']):
#     data = df_['delay_at_checkout_in_minutes'].to_numpy()
#     data = data[~np.isnan(data)]
#     checkout_distrib2[item] = [
#         np.sum(data >= delay_vals, axis=1) / len(data),
#         np.sum(data <= -delay_vals, axis=1) / len(data),
#         ]

##
# df_ = pd.concat([df2['state'] == 'canceled', df2.loc[:, ['checkin_type']]], axis=1)
# df2_ = df.loc[:, ['delay_at_checkout_in_minutes', 'checkin_type']]
# df2_.assign(is_canceled=(df2['state'] == 'canceled'))

delay_vals = np.concatenate([-np.logspace(4, 0, 21), np.logspace(0, 4, 21)])
cancel_prob = {}
for (checkin,), df_ in df2.groupby(['checkin_type']):
    data = np.zeros_like(delay_vals)
    for i, val in enumerate(delay_vals):
        loc = (df_['delay_at_checkout_in_minutes'] >= val)
        # print(df_.loc[loc, 'is_canceled'].mean())
        data[i] = df_.loc[loc, 'is_canceled'].mean()
    cancel_prob[checkin] = data



#%%

##
fig3, axs3 = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4), dpi=200,
    gridspec_kw={'left': 0.075, 'right': 0.97, 'top': 0.85, 'bottom': 0.13,
                 'wspace': 0.18})
fig3.suptitle('Figure 1: checkout delay distribution', x=0.02, ha='left')

labels = [r'$P\,(\mathrm{delay} \geq \tau)$', r'$P\, (\mathrm{delay} \leq - \tau)$',
          'a', 'b']

axs3[0].set_title("Mobile checkin")
line0, = axs3[0].plot(delay_vals, cancel_prob['mobile'],
                      linestyle='-', marker='o', markersize=6, color='tab:blue')
line1, = axs3[0].plot(delay_vals, cancel_prob['mobile'],
             linestyle='-', marker='o', markersize=6, color='tab:orange')
# line2, = axs3[0].plot(delay_vals, checkout_distrib2[('mobile', 'ended')][0],
#                       linestyle='-', marker='v', markersize=6, color='tab:blue')
# line3, = axs3[0].plot(delay_vals, checkout_distrib2[('mobile', 'ended')][1],
#              linestyle='-', marker='v', markersize=6, color='tab:orange')
# axs3[0].text(0.035, 0.06, r'$P\,(\mathrm{delay} \geq 0) = '
#              fr'{checkout_distrib["mobile"][0][0]:.3f}$',
#              transform=axs3[0].transAxes, fontsize=9,
#              bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '0.92'})
axs3[0].grid(visible=True, linewidth=0.3)
# axs3[0].set_xscale('log')
axs3[0].set_xlim(-1e4, 1e4)
# axs3[0].set_yscale('log')
axs3[0].set_ylim(0, 1)
axs3[0].set_xlabel(r"Checkout delay $\tau$ (min)")
axs3[0].set_ylabel('Inverse cumm. prob.')
# axs3[0].legend(handles=[line0, line1, line2, line3], labels=labels)


axs3[1].set_title("Getaround connect checkin")
line0, = axs3[1].plot(delay_vals, cancel_prob['connect'],
                      linestyle='-', marker='o', markersize=6, color='tab:blue')
line1, = axs3[1].plot(delay_vals, cancel_prob['connect'],
                      linestyle='-', marker='o', markersize=6, color='tab:orange')
# line2, = axs3[1].plot(delay_vals, checkout_distrib2[('connect', 'ended')][0],
#                       linestyle='-', marker='v', markersize=6, color='tab:blue')
# line3, = axs3[1].plot(delay_vals, checkout_distrib2[('connect', 'ended')][1],
#                       linestyle='-', marker='v', markersize=6, color='tab:orange')
# axs3[1].text(0.035, 0.06, r'$P\,(\mathrm{delay} \geq 0) = '
#              fr'{checkout_distrib["connect"][0][0]:.3f}$',
#              transform=axs3[1].transAxes, fontsize=9,
#              bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '0.92'})
axs3[1].grid(visible=True, linewidth=0.3)
axs3[1].set_xscale('log')
axs3[1].set_xlim(1, 1e4)
# axs3[1].set_yscale('log')
axs3[1].set_ylim(0, 1)
axs3[1].set_xlabel(r"Checkout delay $\tau$ (min)")
# axs3[1].legend(handles=[line0, line1, line2, line3], labels=labels)


plt.show()






