# -*- coding: utf-8 -*-
"""
Script for the rental delay analysis component of Getaround project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %% Loading
"""
## <a id="loading"></a> Data loading
"""

##
df = pd.read_excel('../data/get_around_delay_analysis.xlsx')
df

print(df.describe(include='all'))

"""
The dataset contains 21310 observations, each consisting of data pertaining to a car rental event.
The dataset has 7 columns:
- The column `car_id` refers to the car that was rented. In the absence of further information, it is of no use to us.
- The columns `rental_id` and `previous_ended_rental_id` are identifiers of the current and previous rentals of a given car.
We will use them to follow car rental sequences.
- The column `checkin_type` indicates whether the rental was made using Getaround connect functionality or by mobile.
- The column `state` indicates whether the rental was canceled or not.
- The column `delay_at_checkout_in_minutes` gives the time difference between the actual and expected checkout times.
A negative value indicates that the checkout occured earlier than expected, and a positive value indicates a late checkout. 
A late checkout which makes the next customer waiting is problematic and this is what we aim to mitigate by introducing a delay
before availability.
- The column `time_delta_with_previous_rental_in_minutes` represents the expected amount of time between two consecutive rentals.
This value is based on the *expected* checkout and checkin times, and does not include the checkout delay.
A `NULL` value corresponds to a time delta larger that 12h (720 min), in which case the rental is assumed to be non-consecutive
(`previous_ended_rental_id` is also `NULL`).
"""

# %% EDA
"""
## <a id="eda"></a> Exploratory data analysis

Before determining the impact of the introduction of a rental delay, we first gather some
necessary insights about user behavior.
"""

## Number of rentals using each method
print(df['checkin_type'].value_counts())

## Counts of rental states for each checkin type
df_ = df.groupby(['checkin_type', 'state']).count()['rental_id']
print(df_)

## Probability of rental states for each checkin type
print(df_ / df_.T.groupby('checkin_type').sum())

"""
- Customers favor mobile checkin (80%) over Getaround connect (20%).
Part of this difference is due to the fact that not all the cars (actually, only 46%)
have the Getaround connect option.
- Rental cancellation rates are higher when customers use Getaround connect functionality (18.5%)
than with mobile checkin (14.5%). The cancellation process is possibly made easier with Getaround connect.
"""

# %%
"""
### Checkout delay


"""

##
df_ = df.groupby(['checkin_type', 'state']).agg(lambda x: x.isnull().sum())
print(df_)

"""
Almost all null `delay_at_checkout_in_minutes` when canceled
"""


##
delay_vals = np.logspace([0], [5], 21)
checkout_distrib = {}
avg_checkout_delay, median_checkout_delay = {}, {}
for (checkin,), df_ in df.groupby(['checkin_type']):
    avg_checkout_delay[checkin] = df_['delay_at_checkout_in_minutes'].mean()
    median_checkout_delay[checkin] = df_['delay_at_checkout_in_minutes'].median()
    data = df_['delay_at_checkout_in_minutes'].to_numpy()
    data = data[~np.isnan(data)]
    checkout_distrib[checkin] = [
        np.sum(data >= delay_vals, axis=1) / len(data),
        np.sum(data <= -delay_vals, axis=1) / len(data),
        ]

## summary text to display on the figure
summary_text = {k: (f'avg delay = {avg_checkout_delay[k]:.0f} min\n'
                    f'median delay = {median_checkout_delay[k]:.0f} min\n'
                    r'$P\,(\mathrm{delay} \geq 0) '
                    f'= {checkout_distrib[k][0][0]:.3f}$')
                for k in checkout_distrib}

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
axs1[0].text(0.035, 0.06, summary_text['mobile'],
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
axs1[1].text(0.035, 0.06, summary_text['connect'],
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
Figure 1 presents the inverse cummulative distributions of checkout delays,
for both mobile checkin (left panel) and Getaround connect checkin (right panel).

Delays are shorter and occur less frequently with Getaround connect.
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
    gridspec_kw={'left': 0.07, 'right': 0.93, 'top': 0.85, 'bottom': 0.13,
                 'wspace': 0.24})
axs2_twin = [ax.twinx() for ax in axs2]
fig2.suptitle('Figure 2: Distribution of delays with previous rental', x=0.02, ha='left')


handles = [Patch(facecolor='tab:blue', alpha=1, label='ended'),
           Patch(facecolor='tab:orange', alpha=1, label='canceled')]
labels = ['ended', 'canceled', 'cancelation prob.']

axs2[0].set_title("Mobile checkin")
axs2[0].hist(rental_delay['mobile'], bins=np.linspace(-30, 750, 14),
             stacked=False, density=False)

line, = axs2_twin[0].plot(delay_vals, delay_hist_mfrac, color='tab:red',
                  marker='o', markersize=5)
axs2_twin[0].set_ylim(0, 0.425)
axs2_twin[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axs2_twin[0].set_yticks([0.05, 0.15, 0.25, 0.35], minor=True)
# axs2_twin[0].set_ylabel('Cancellation prob.', rotation=270, labelpad=12)

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

line, = axs2_twin[1].plot(delay_vals, delay_hist_cfrac, color='tab:red',
                          marker='o', markersize=5)
axs2_twin[1].set_ylim(0, 0.425)
axs2_twin[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axs2_twin[1].set_yticks([0.05, 0.15, 0.25, 0.35], minor=True)
axs2_twin[1].set_ylabel('Cancellation prob.', rotation=270, labelpad=14)

axs2[1].grid(visible=True, linewidth=0.3)
axs2[1].set_xlim(-30, 750)
axs2[1].set_xticks(np.linspace(0, 720, 7))
axs2[1].set_xticks(np.linspace(60, 660, 6), minor=True)
axs2[1].set_xlabel("Delay with previous rental (min)")
axs2[1].set_ylim(0, 170)
# axs2[1].set_ylabel("Counts")
axs2[1].legend(handles=handles + [line], labels=labels)


plt.show()

"""
Figure 2

Cancellation is due to the checkout later than the rental delay.
"""

# %%

"""
### Cancel probability

We now study the rental cancellation probability as a function of the delay at checkout.
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


##
delay_vals_neg, delay_vals_pos = -np.logspace(4, 0, 21), np.logspace(0, 4, 21)
delay_vals = np.concatenate([delay_vals_neg, delay_vals_pos])
cancel_prob = {}
cancel_prob_std = {}
for (checkin,), df_ in df2.groupby(['checkin_type']):
    prob = np.zeros_like(delay_vals)
    prob_std =  np.zeros_like(delay_vals)
    for i, val in enumerate(delay_vals):
        loc = (df_['delay_at_checkout_in_minutes'] >= val)
        p = df_.loc[loc, 'is_canceled'].mean()
        prob[i] = p
        prob_std[i] = p*(1-p) / np.sqrt(df_.loc[loc, 'is_canceled'].sum())
        
    cancel_prob[checkin] = prob
    cancel_prob_std[checkin] = prob_std

##
p_cancel = df2[['checkin_type', 'is_canceled']].groupby('checkin_type').mean()
p_cancel

"""
The baseline cancel probability is larger with getaround connect.
"""

#%%

##
fig3, axs3 = plt.subplots(
    nrows=1, ncols=2, sharey=True, figsize=(6, 3.8), dpi=200,
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.88, 'bottom': 0.13,
                 'wspace': 0.025})
fig3.suptitle('Figure 3: Cumulative cancellation probability', x=0.02, ha='left')


axs3[0].spines.right.set_visible(False)
axs3[0].plot([1, 1], [0, 1], transform=axs3[0].transAxes,
             marker=[(-0.4, -1), (0.4, 1)], markersize=10, linestyle='',
             color='k', mec='k', mew=1, clip_on=False)
axs3[1].spines.left.set_visible(False)
axs3[1].plot([0, 0], [0, 1], transform=axs3[1].transAxes,
             marker=[(-0.4, -1), (0.4, 1)], markersize=10, linestyle='',
             color='k', mec='k', mew=1, clip_on=False)
axs3[1].tick_params(axis='y', labelleft=False, left=False, which='both')
ms = 5

eb0 = axs3[0].errorbar(delay_vals_pos, cancel_prob['mobile'][20::-1],
                       yerr=cancel_prob_std['mobile'][20::-1],
                       linestyle='-', marker='o', markersize=ms, color='tab:blue')
eb1 = axs3[0].errorbar(delay_vals_pos, cancel_prob['connect'][20::-1],
                       yerr=cancel_prob_std['connect'][20::-1],
                      linestyle='-', marker='o', markersize=ms, color='tab:orange')
axs3[0].grid(visible=True, linewidth=0.3)
axs3[0].grid(visible=True, axis='y', which='minor', linewidth=0.2)
axs3[0].invert_xaxis()
axs3[0].set_xscale('log')
axs3[0].set_xlim(1e4, 0.8)
axs3[0].set_xticks([1e4, 1e3, 1e2, 1e1, 1],
                     ['-$10^4$', '-$10^3$', '-$10^2$', '-$10^1$', '-1'])
axs3[0].set_ylim(0, 1)
axs3[0].set_yticks(np.linspace(0.1, 0.9, 5), minor=True)
axs3[0].set_ylabel('Inverse cumm. prob.', fontsize=11)
axs3[0].legend(handles=[eb0, eb1],
               labels=['Mobile checkin', 'Getaround connect checkin'],
               loc=(0.02, 0.82))


axs3[1].errorbar(delay_vals_pos, cancel_prob['mobile'][21:],
                 yerr=cancel_prob_std['mobile'][21:],
                 linestyle='-', marker='o', markersize=ms, color='tab:blue')
axs3[1].errorbar(delay_vals_pos, cancel_prob['connect'][21:],
                 yerr=cancel_prob_std['connect'][21:],
                 linestyle='-', marker='o', markersize=ms, color='tab:orange')
axs3[1].grid(visible=True, linewidth=0.3)
axs3[1].grid(visible=True, axis='y', which='minor', linewidth=0.2)
axs3[1].set_xscale('log')
axs3[1].set_xlim(0.8, 1e4)
axs3[1].set_xticks([1, 1e1, 1e2, 1e3, 1e4],
                     ['1', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])

fig3.text(0.53, 0.025, r"Checkout delay $\tau$ (min)",
          fontsize=11, ha='center')


plt.show()

"""
Figure 3 presents the cumulative probability of rental cancellation.

as a function of the previous rental checkout delay.
$\tau$
"""

# %%
"""
## <a id="delay"></a> Introducing a delay between rentals

### loss of revenue vs delay
"""

## Get an estimatin of the average rental prices
df_ = pd.read_csv('../data/get_around_pricing_project.csv')
loc = df_['has_getaround_connect']
rental_price = {'mobile': df_['rental_price_per_day'].mean(),
                'connect': df_.loc[loc]['rental_price_per_day'].mean()}

## Construct the dataframe for revenue loss estimation
prev_rental_cols = ['rental_id', 'delay_at_checkout_in_minutes']
curr_rental_cols = ['previous_ended_rental_id', 'checkin_type', 'state',
                    'time_delta_with_previous_rental_in_minutes']
df_prev = df.loc[:, prev_rental_cols]
df_curr = df.loc[:, curr_rental_cols]
df_loss = pd.merge(df_prev, df_curr, how='inner', left_on='rental_id',
                   right_on='previous_ended_rental_id')
df_loss = df_loss.assign(is_canceled=(df_loss['state'] == 'canceled'))


def revenue_loss(delay: float):
    # TODO
    info = {}
    for (scope,), df_ in df_loss.groupby('checkin_type'):
        loc = (df_['delay_at_checkout_in_minutes'] >= delay)
        
        p = df_.loc[loc, 'is_canceled'].mean()
        n = df_.loc[loc].sum()
        info[scope] = {
            'cancel_prob': p,
            'cancel_nb': n*p,
            'revenue_loss': n*(1-p) * rental_price[scope],
            'revenue_loss_per_car': (1-p) * rental_price[scope],
            }
    
    return info


