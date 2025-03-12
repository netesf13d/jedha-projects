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

cancelled_checkout_delay = 0
ended_checkout_delay = 0

checkout_delay = {}
for state, df_ in df.groupby('state'):
    checkout_delay[state] = df_.loc[:, ['checkin_type', 'delay_at_checkout_in_minutes']]



# %%

# cols = ['checkin_type', 'state', 'delay_at_checkout_in_minutes',
#        'time_delta_with_previous_rental_in_minutes']

# data = df.groupby(['checkin_type', 'state']).count()['car_id'].to_numpy()



fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4), dpi=200,
    gridspec_kw={'left': 0.06, 'right': 0.97, 'top': 0.85, 'bottom': 0.13,
                 'wspace': 0.18})
fig1.suptitle('Figure 1: ', x=0.02, ha='left')


axs1[0].set_title("Checkout delay distribution")
axs1[0].hist(df['delay_at_checkout_in_minutes'], bins=np.linspace(-100, 100, 22))
axs1[0].grid(visible=True, linewidth=0.3)
# axs1[0].set_xlim(-15, 735)
axs1[0].set_xlabel("Checkout delay (years)")
axs1[0].set_ylabel('Counts')
# axs1[0].text(0.575, 0.84, textstr(sufrom mpl_toolkits.axes_grid1 import make_axes_locatablebject_df['age'].to_numpy()),
#              transform=axs1[0].transAxes, fontsize=9,
#              bbox={'boxstyle': 'round', 'facecolor': '0.92'})


axs1[1].set_title("Delay with previous rental")
axs1[1].hist(df['time_delta_with_previous_rental_in_minutes'],
             bins=np.linspace(-15, 735, 26))
axs1[1].grid(visible=True, linewidth=0.3)
axs1[1].set_xlim(-15, 735)
axs1[1].set_xticks(np.linspace(0, 720, 7))
axs1[1].set_xticks(np.linspace(60, 640, 6), minor=True)
axs1[1].set_xlabel("Delay with previous rental")


# cmap1 = plt.get_cmap('plasma').copy()
# cmap1.set_extremes(under='0.7', over='0.5')
# axs1[2].set_aspect('equal')

# heatmap = axs1[2].pcolormesh(data.reshape((2, 2))/np.sum(data), linewidth=0.6,
#                              edgecolors='0.3', cmap=cmap1, vmin=0, vmax=1)
# axs1[2].set_xticks([0.5, 1.5], ['cancelled', 'ended'])
# axs1[2].set_yticks([0.5, 1.5], ['connect', 'mobile'])
# for (i, j), val in np.ndenumerate(data.reshape((2, 2))):
#     axs1[2].text(0.5+i, 0.5+j, val, ha='center', va='center')

# div = make_axes_locatable(axs1[2])
# axs1_cb = div.append_axes('top', size="7%", pad="7%")
# fig1.add_axes(axs1_cb)
# fig1.colorbar(heatmap, cax=axs1_cb, orientation="horizontal", ticklocation='top')

plt.show()


# %%










