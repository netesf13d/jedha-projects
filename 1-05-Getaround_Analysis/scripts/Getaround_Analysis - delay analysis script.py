# -*- coding: utf-8 -*-
"""
Script for Getaround project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



# %% Loading
"""
## <a name="loading"></a> Data loading
"""

##
pp_df = pd.read_csv('../data/get_around_pricing_project.csv')
pp_df = pp_df.drop('Unnamed: 0', axis=1)
pp_df

##
da_df = pd.read_excel('../data/get_around_delay_analysis.xlsx')
da_df

"""
The dataset is split in two parts, with 21310 observations each.
- The first dataframe, `pp_df`, contains the information that we have *before* a car is rent.
We will use these features as predictors for our machine learning model
- The second dataframe contains information relative to the *outcome* of a rental. This data cannot be used to make predictions.
However, we will use it to build target feature to train our ML model.

We should include all the data available in our exploratory data analysis. we build a combined dataframe `eda_df` for this purpose.
"""

eda_df = pd.concat([da_df, pp_df], axis=1)
eda_df


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

##
a = da_df['car_id'].astype(object).describe()
print(a)

##
b = da_df.loc[:, ['delay_at_checkout_in_minutes', 'time_delta_with_previous_rental_in_minutes']].describe()
print(b)

##
c = da_df.loc[:, ['checkin_type', 'state']].describe()
print(c)


"""
There are 8143 cars available on the platform
"""

# %%
"""
### some graphs


- follow the trajectory of a car
"""

cols = ['private_parking_available', 'has_gps', 'has_air_conditioning',
        'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
        'winter_tires']
df_ = eda_df.groupby('car_id').first()


#%%

fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(7, 4.2), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.975, 'top': 0.91, 'bottom': 0.11, 'hspace': 0.08})
fig1.suptitle("Figure 1: Comparison of rental prices vs various car attributes",
              x=0.02, ha='left')


vdata, vpos = [], []
vdata_true, vdata_false = [], []
for i, item in enumerate(cols):
    for is_, gdf in pp_df.groupby(item):
        if is_:
            vdata_true.append(gdf['rental_price_per_day'].to_numpy())
        else:
            vdata_false.append(gdf['rental_price_per_day'].to_numpy())

vplot_false = ax1.violinplot(vdata_false, positions=np.linspace(-0.2, i-0.2, i+1),
                            widths=0.3, showmeans=True)
for elt in vplot_false['bodies']:
    elt.set_facecolor('#1F77B480')
for elt in ['cbars', 'cmaxes', 'cmins', 'cmeans']:
    vplot_false[elt].set_edgecolor('tab:blue')

vplot_true = ax1.violinplot(vdata_true, positions=np.linspace(0.2, i+0.2, i+1),
                            widths=0.3, showmeans=True)
for elt in vplot_true['bodies']:
    elt.set_facecolor('#FF7F0E80')
for elt in ['cbars', 'cmaxes', 'cmins', 'cmeans']:
    vplot_true[elt].set_edgecolor('tab:orange')


ax1.set_xlim(-0.5, i+0.5)
ticks = ['Private\nparking', 'GPS', 'Air\nConditioning', 'Automatic\nCar',
         'Getaround\nConnect', 'Speed\nRegulator', 'Winter\nTires']
ax1.set_xticks(np.arange(0, len(ticks)), ticks, rotation=0)
ax1.set_ylim(0, 480)
ax1.set_yticks(np.arange(50, 500, 50), minor=True)
ax1.set_ylabel('Rental price ($/day)', labelpad=5)
ax1.grid(visible=True, axis='y', linewidth=0.3)
ax1.grid(visible=True, axis='y', which='minor', linewidth=0.2)

handles = [Patch(facecolor='#1F77B480', edgecolor='k', linewidth=0.5),
           Patch(facecolor='#FF7F0E80', edgecolor='k', linewidth=0.5)]
ax1.legend(handles=handles, labels=['False', 'True'], ncols=2)

plt.show()

"""
Figure 1 presents violin plots of car rental price as a function of various car options.
The presence of an option, such as a GPS or private parking affects the price positively.
"""

# %% ML










