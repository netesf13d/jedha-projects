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
## <a id="eda_user"></a> Study of user behavior

Before determining the impact of the introduction of a rental delay, we first gather some
necessary insights about user behavior.


### General user behavior
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

## Number of NULL values
df_ = df.groupby(['checkin_type', 'state']).agg(lambda x: x.isnull().sum())
print(df_)

"""
- Almost all `delay_at_checkout_in_minutes` are `NULL` when the rental was canceled
(save for 1 value which is probably an error). CHeckout never occurs when the renatl is canceled.
- Even when the rental ended, the delay at checkout is sometimes unknown. This happens about
10% of the time with mobile checkin, but less than 3% with getaround connect.
"""

## Fraction of rentals which are consecutive (ie with less than 12h between checkout and next checkin)
df_ = df.groupby('checkin_type').agg(lambda x: (~x.isnull()).sum())['previous_ended_rental_id']
print(df_ / df['checkin_type'].value_counts())

"""
Consecutive rentals are much more frequent when customers use Getaround connect functionality (19%)
than with mobile checkin (6%). Getaround connect is certainly beneficial to the company as it reduces
the time spent in the un-rented state.
"""

# %%
r"""
### Distribution of checkout delays

In this section, we study the distribution of checkout delays. The range of checkout delays
is extremely broad, ranging from -22433 min (about 16 days!) to 71084 min (more than 49 days!).
To visualize clearly the distribution of checkout delays, we compute the complementary cumulative
distributions of delays. For a given time delay $\tau$ the positive and negative complementary
cumulative distributions are respectively:
$$
    \mathrm{Prob}\, \left( T \geq \tau \right), \quad \mathrm{Prob}\left( T \leq -\tau \right),
$$
where $T$ is the checkout delay.
"""

## Compute the complementary cumulative distribution of checkout delays
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
fig1.suptitle('Figure 1: Complementary cumulative distribution of checkout delays',
              x=0.02, ha='left')

labels = [r'$\mathrm{Prob}\,(T \geq \tau)$', r'$\mathrm{Prob}\, (T \leq - \tau)$']

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
Figure 1 presents the complementary cummulative distributions of checkout delays,
for both mobile checkin (left panel) and Getaround connect checkin (right panel).
Most delays are rather short, but delays larger than 12 hours are not infrequent,
they occur about 10% of the time.
However, there is a significant difference between the two checkin methods.
Delays tend to be shorter and occur less frequently with Getaround connect. Moreover,
delays larger than a day never occur with Getaround connect.
"""


# %%
"""
### Delay with previous rental

We turn to the analysis of the delays between consecutive rentals. We recall that
these events are defined rental delays less than 720 min (12 hours),
and that they account for only 19% of the cases with Getaround connect
and 6% of the cases with mobile checkin.
"""

## Rental delay values
rental_delay = {'mobile': [0, 0], 'connect': [0, 0]}
for (checkin, state), df_ in df.groupby(['checkin_type', 'state']):
    if state == 'ended':
        rental_delay[checkin][0] = df_['time_delta_with_previous_rental_in_minutes'].to_numpy()
    if state == 'canceled':
        rental_delay[checkin][1] = df_['time_delta_with_previous_rental_in_minutes'].to_numpy()

## histogram bins and center values
delay_bins = np.linspace(-30, 750, 14)
delay_vals = (delay_bins[1:] + delay_bins[:-1]) / 2

## 'mobile' delay histograms : ended, canceled, cancelation prob, prob stddev
delay_hist_me, _ = np.histogram(rental_delay['mobile'][0], bins=delay_bins)
delay_hist_mc, _ = np.histogram(rental_delay['mobile'][1], bins=delay_bins)
delay_mfrac = delay_hist_mc / (delay_hist_mc + delay_hist_me)
delay_mfrac_std = delay_mfrac * (1 - delay_mfrac) / np.sqrt(delay_hist_mc + delay_hist_me)

## 'connect' delay histograms : ended, canceled, cancelation prob, prob stddev
delay_hist_ce, _ = np.histogram(rental_delay['connect'][0], bins=delay_bins)
delay_hist_cc, _ = np.histogram(rental_delay['connect'][1], bins=delay_bins)
delay_cfrac = delay_hist_cc / (delay_hist_cc + delay_hist_ce)
delay_cfrac_std = delay_cfrac * (1 - delay_cfrac) / np.sqrt(delay_hist_cc + delay_hist_ce)


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

err = axs2_twin[0].errorbar(delay_vals, delay_mfrac, delay_mfrac_std,
                            color='tab:red', marker='o', markersize=5)
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
axs2[0].legend(handles=handles + [err], labels=labels)


axs2[1].set_title("Getaround connect checkin")
axs2[1].hist(rental_delay['connect'], bins=np.linspace(-30, 750, 14), # np.linspace(-15, 735, 26)
             stacked=False, density=False)

err = axs2_twin[1].errorbar(delay_vals, delay_cfrac, delay_cfrac_std,
                            color='tab:red', marker='o', markersize=5)
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
axs2[1].legend(handles=handles + [err], labels=labels)


plt.show()

"""
We show in figure 2 histograms of the delay with previous rental, for both
mobile checkin (left panel) and Getaround connect checkin (right panel),
distinguishing ended and caceled rentals. We also show the associated cancellation probability.
The histograms are binned with hourly intervals.
There is a dip at around 6h rental delay. This is likely a consequence of user car checkout and checkin schedule.
Users tend to checkout late in the day and do not checkin at night.

We note that the counts are similar for both checkin methods despite the fact that mobile
checkins are 4 times more frequent. We recover the fact that Getaround connect functionality
favors consecutive rentals. We also observe the higher probability of cancellation
with getaround checkin mentioned above. Interestingly, this cancellation probability seems
independent of the rental delay.
"""

# %%

"""
## <a id="eda_rentals"></a> Study of consecutive rentals

We now focus on consecutive rentals. In this section we study how the checkout delay of the
previous rental affects the subsequent rental. For that purpose, we create a new dataframe
by joining the relevant elements that have matching `rental_id` and `previous_ended_rental_id`.
"""

## Merge checkout delay of previous rental with characteristics of subsequent rental
prev_rental_cols = ['rental_id', 'delay_at_checkout_in_minutes']
curr_rental_cols = ['previous_ended_rental_id', 'checkin_type', 'state',
                    'time_delta_with_previous_rental_in_minutes']
df_prev = df.loc[:, prev_rental_cols]
df_curr = df.loc[:, curr_rental_cols]
df2 = pd.merge(df_prev, df_curr, how='inner', left_on='rental_id',
               right_on='previous_ended_rental_id')
df2 = df2.assign(is_canceled=(df2['state'] == 'canceled'))

print(df2.describe(include='all'))


"""
### Cancellation probability vs checkout delay

We now study the rental cancellation probability as a function of the delay
at checkout of the previous rental. We are still facing the problem of broad
checkout delay ranges. For a given time delay $\tau$, we are interested in
the cancellation probability when the checkout delay $T$ is greater than $\tau$:
$$
    \mathrm{Prob}\, \left(\mathrm{cancel} \,|\, T \geq \tau \right).
$$
"""

## Bseline cancellation probability for consecutive rentals
p_cancel = df2[['checkin_type', 'is_canceled']].groupby('checkin_type').mean()
print(p_cancel)

"""
This quantity represents the baseline cancel probability for consecutive rentals or,
with the previous expression, $\mathrm{Prob}\left(\mathrm{cancel} | T \geq -\infty \right)$.
The value is still larger for Getaround connect (16%) than for mobile checkin (9%).
However, the difference is reduced as compared to the global cancellation probability
including non-consecutive rentals (19% vs 6%, respectively).
This means that Getaround connect functionality actually helps mitigating the cancellation issue of consecutive rentals.
"""


## Compute the cumulative cancellation probability vs checkout delay
delay_vals_neg, delay_vals_pos = -np.logspace(4, 0, 21), np.logspace(0, 4, 21)
delay_vals = np.concatenate([delay_vals_neg, delay_vals_pos])
cancel_prob = {}
cancel_prob_std = {}
for (checkin,), df_ in df2.groupby(['checkin_type']):
    prob = np.zeros_like(delay_vals)
    prob_std = np.zeros_like(delay_vals)
    for i, val in enumerate(delay_vals):
        loc = (df_['delay_at_checkout_in_minutes'] >= val)
        p = df_.loc[loc, 'is_canceled'].mean()
        prob[i] = p
        prob_std[i] = p*(1-p) / np.sqrt(df_.loc[loc, 'is_canceled'].sum())

    cancel_prob[checkin] = prob
    cancel_prob_std[checkin] = prob_std


##
fig3, axs3 = plt.subplots(
    nrows=1, ncols=2, sharey=True, figsize=(6, 3.8), dpi=200,
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.88, 'bottom': 0.13,
                 'wspace': 0.025})
fig3.suptitle('Figure 3: Cumulative cancellation probability '
              r'$\mathrm{Prob}\, \left(\mathrm{cancel} \,|\, T \geq \tau \right)$',
              x=0.02, ha='left')


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
Figure 3 presents the cumulative probability of rental cancellation,
\mathrm{Prob}\, \left(\mathrm{cancel} \,|\, T \geq \tau \right), for both
checkin methods. The points at large negative delay $\tau$ correspond (up to fluctuations)
to the baseline cancellation probability. The curve is roughly constant until $\tau$
becomes positive. At this point, events with large checkout delay start
to contribute, and we observe a rise in cancellation probability
for both checkin methods.
However, there is a significant difference between the two checkin methods.
Although mobile checkin cancellation reaches a maximum of about 20%, the cancellation probability
rises sharply with delays larger than 100 min with Getaround connect method.
"""

# %%
"""
### Cancellation probability vs customer waiting time

!!!
We posit that the cancellation of a rental is mostly due to !!!

the checkout later than the rental delay.

This time, we keep our objective in mind of analyzing the effects of a rental delay.
We therefore compute the cancellation probability on a linear scale rather than logarithmic scale.
"""

## Waiting time: difference between checkout delay and rental delay
## Positive when the next customer had to wait
waiting_time = (df2['delay_at_checkout_in_minutes']
                - df2['time_delta_with_previous_rental_in_minutes'])
df2 = df2.assign(waiting_time_in_minutes=waiting_time)

df2['waiting_time_in_minutes'].describe()



#%%

## Cancellation probability
waiting_times = np.linspace(-500, 500, 51)
cancel_prob2 = {}
cancel_prob2_std = {}
for (checkin,), df_ in df2.groupby(['checkin_type']):
    prob = np.zeros_like(waiting_times)
    prob_std =  np.zeros_like(waiting_times)
    for i, val in enumerate(waiting_times):
        loc = (df_['waiting_time_in_minutes'] >= val)
        p = df_.loc[loc, 'is_canceled'].mean()
        prob[i] = p
        prob_std[i] = p*(1-p) / np.sqrt(df_.loc[loc, 'is_canceled'].count())

    cancel_prob2[checkin] = prob
    cancel_prob2_std[checkin] = prob_std


##
fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, figsize=(6, 4), dpi=200,
    gridspec_kw={'left': 0.1, 'right': 0.97, 'top': 0.9, 'bottom': 0.13,
                 'wspace': 0.24})
fig4.suptitle('Figure 4: !!!',
              x=0.02, ha='left')


ax4.errorbar(waiting_times, cancel_prob2['mobile'], cancel_prob2_std['mobile'],
                    color='tab:blue', marker='o', markersize=5,
                    label='Mobile checkin')
ax4.errorbar(waiting_times, cancel_prob2['connect'], cancel_prob2_std['connect'],
                    color='tab:orange', marker='o', markersize=5,
                    label='Getaround connect checkin')

ax4.grid(visible=True, linewidth=0.3)
# ax4.set_xlim(-30, 750)
# ax4.set_xticks(np.linspace(0, 720, 7))
# ax4.set_xticks(np.linspace(60, 660, 6), minor=True)
ax4.set_xlabel("Waiting time (min)")
ax4.set_ylim(0, 1)
ax4.set_ylabel("Counts")
ax4.legend()

plt.show()

"""
!!!
Figure 4 

"""


# %%
"""
## <a id="delay"></a> Introducing a delay between rentals


### Probability density

"""

## Probability density estimation bins and center values
time_bins = np.linspace(-750, 750, 26)
# time_bins = np.concatenate([np.linspace(-750, -210, 10),
#                             np.linspace(-180, 180, 13),
#                             np.linspace(210, 750, 10)])

time_vals = (time_bins[1:] + time_bins[:-1]) / 2

## Probabilities
p_cancel = {} # cancellation probability
std_p_cancel = {} # std dev estimate of cancellation probability
pd_waiting_time = {} # waiting time probability density
for (checkin,), df_ in df2.groupby(['checkin_type']):
    # the two extremal points correspond to values outside the bins
    prob = np.zeros(len(time_vals)+2, dtype=float)
    std_prob = np.zeros(len(time_vals)+2, dtype=float)
    counts = np.zeros(len(time_vals)+2, dtype=int)

    df_ = df_.loc[~df_['waiting_time_in_minutes'].isna()]
    idx = np.digitize(df_['waiting_time_in_minutes'], time_bins)
    for i in range(len(time_bins) + 1):
        temp = df_.iloc[idx == i]['is_canceled']
        counts[i] = temp.count()
        prob[i] = temp.mean()
        std_prob[i] = prob[i]*(1-prob[i]) / np.sqrt(counts[i])

    p_cancel[checkin] = prob
    std_p_cancel[checkin] = std_prob
    pd_waiting_time[checkin] = counts / np.sum(counts)


# %%
##
fig5, axs5 = plt.subplots(
    nrows=2, ncols=1, figsize=(6.4, 6), dpi=200,
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.92, 'bottom': 0.08,
                 'hspace': 0.28})
fig5.suptitle('Figure 5: !!!', x=0.02, ha='left')


x = 790 # position out of range probabilities

axs5[0].errorbar(time_vals, p_cancel['mobile'][1:-1], std_p_cancel['mobile'][1:-1],
                       color='tab:blue', marker='o', markersize=4,
                       label='Mobile checkin')
axs5[0].errorbar([-x, x], p_cancel['mobile'][[0, -1]], std_p_cancel['mobile'][[0, -1]],
             color='tab:blue', linestyle='', marker='s', markersize=6)

axs5[0].errorbar(time_vals, p_cancel['connect'][1:-1], std_p_cancel['connect'][1:-1],
                       color='tab:orange', marker='o', markersize=4,
                       label='Getaround connect checkin')
axs5[0].errorbar([-x, x], p_cancel['connect'][[0, -1]], std_p_cancel['connect'][[0, -1]],
             color='tab:orange', linestyle='', marker='s', markersize=6)

axs5[0].grid(visible=True, linewidth=0.3)
axs5[0].set_xlim(-800, 800)
axs5[0].set_xlabel('Waiting time (min)', fontsize=11)
axs5[0].set_ylim(-0.01, 1.01)
axs5[0].set_ylabel("Cancellation probability", fontsize=11, labelpad=7)
axs5[0].legend(loc=(0.01, 0.67), title='Cancellation probability',
               title_fontproperties={'weight': 'bold'})


axs5[1].plot(time_vals, pd_waiting_time['mobile'][1:-1],
             color='tab:blue', marker='o', markersize=4,
             label='Mobile checkin')
axs5[1].plot([-x, x], pd_waiting_time['mobile'][[0, -1]],
             color='tab:blue', linestyle='', marker='s', markersize=6)

axs5[1].plot(time_vals, pd_waiting_time['connect'][1:-1],
             color='tab:orange', marker='o', markersize=4,
             label='Getaround connect checkin')
axs5[1].plot([-x, x], pd_waiting_time['connect'][[0, -1]],
             color='tab:orange', linestyle='', marker='s', markersize=6)

axs5[1].grid(visible=True, linewidth=0.3)
axs5[1].grid(visible=True, which='minor', linewidth=0.2)
axs5[1].set_xlim(-800, 800)
axs5[1].set_xlabel('Waiting time (min)', fontsize=11)
axs5[1].set_ylim(-0.002, 0.3)
axs5[1].set_yticks(np.linspace(0, 0.3, 4))
axs5[1].set_yticks(np.linspace(0.05, 0.25, 3), minor=True)
axs5[1].set_ylabel("Probability density", fontsize=11, labelpad=7)
axs5[1].legend(loc=(0.01, 0.67), title='Waiting time probability density',
               title_fontproperties={'weight': 'bold'})


plt.show()



# %%
"""
### loss of revenue vs delay

We repeat here the whole data treatment
"""


## Get an estimation of the average rental prices
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


