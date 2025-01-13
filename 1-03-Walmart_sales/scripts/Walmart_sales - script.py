#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script version of the Walmart sales project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# %% Loading
"""
## <a name="loading"></a> Data loading and preprocessing
"""

# Load data
raw_df = pd.read_csv('../Walmart_Store_sales.csv')
raw_df = raw_df.assign(Store=raw_df['Store'].astype(int))
raw_df = raw_df.assign(Date=pd.to_datetime(raw_df['Date'], format='%d-%m-%Y'))
# Remove records with NaNs in target 'Weekly_Sales'
df = raw_df.dropna(subset=['Weekly_Sales'])

df_desc = df.describe()
df_desc


"""
We remove outlier observations in the numeric features `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`.
Are considered as outliers those features values which lie more than 3 standard deviations from the mean.
"""

num_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
means = df_desc.loc['mean', num_features]
stds = df_desc.loc['std', num_features]
a = pd.concat([means - 3*stds, df_desc.loc[['min', 'max'], num_features].T, means + 3*stds],
              axis=1)
print(a)

"""
The lower bound for outlier exclusion is indicated as the column `0`, while the upper bound is the column `1`.
The numerical features are all within the bounds. However, doing a filtering such as
```python
(df[feature] >= mean-3*std) & (df[feature] <= mean+3*std)
```
would return `True` for missing values (NaNs). It would effectively remove those observations which have
at least one missing numerical feature. Doing so reduces the dataset to only 90 observations, and we assume that it is undesired.
We will rather rely on an imputing strategy for the missing values.

Before that, we process the date to convert it into a numerical timestamp.
The weekly sales were aggregated on a weekly basis, every friday.
We therefore extract only the year, month and week number from the date.
# """
##
df = df.assign(Date=pd.to_datetime(df['Date'], format='%d-%m-%Y'))
df['Date'].dt.day_name()

# ##
df.loc[:, 'Year'] = df['Date'].dt.year
df.loc[:, 'Month'] = df['Date'].dt.month
df.loc[:, 'Week'] = df['Date'].dt.isocalendar().week


# %% EDA
"""
## <a name="eda"></a> Preliminary data analysis

#### 

We begin by getting insights about the store locations.
"""

COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]


fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, sharey=False, figsize=(8, 4.8), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.96, 'top': 0.72, 'bottom': 0.15, 'wspace': 0.24})
fig1.suptitle("Figure 1: Time dependence of CPI and temperature",
              x=0.02, ha='left')

handles, labels = [], []
for istore, store_df in df.groupby('Store'):
    # CPI vs date
    line, = axs1[0].plot(store_df['Date'], store_df['CPI'], color=COLORS[istore],
                         marker='o', markersize=4, linestyle='')

    # temperature vs month
    gdf = store_df.groupby('Month').mean()
    axs1[1].plot(gdf.index, gdf['Fuel_Price'], color=COLORS[istore],
                 marker='o', markersize=5, linestyle='')

    # legend handles and labels
    handles.append(line)
    labels.append(f'store {istore}')

dates = pd.date_range(start='1/1/2010', end='1/1/2013', freq='6MS')
axs1[0].set_xticks(dates, dates.strftime('%Y-%m'), rotation=30)
axs1[0].set_xlabel('Date')
axs1[0].set_ylim(120, 240)
axs1[0].set_ylabel('Consumer Price Index')
axs1[0].grid(visible=True, linewidth=0.2)

axs1[1].set_xticks(np.arange(1, 13, 1))
axs1[1].set_xlabel('Month')
# axs1[1].set_ylim(0, 100)
# axs1[1].set_yticks(np.arange(10, 100, 20), minor=True)
# axs1[1].set_ylabel('Temperature (°F)', labelpad=2)
axs1[1].grid(visible=True, linewidth=0.3)
axs1[1].grid(visible=True, which='minor', linewidth=0.2)

fig1.legend(handles=handles, labels=labels,
            ncols=5, loc=(0.14, 0.745), alignment='center')

plt.show()

"""
We show on the left panel of figure 1 the time dependence of the consumer price index for each store.
The observed slow increase over time is due to inflation. We can clearly split the stores into 3 groups
based on the range of CPI values : those probably correspond to stores located in different regions of the world.

The right panels shows the monthly average of the temperature (right panel). For each store,
the temperature follows the north hemisphere seasonal variations.
"""

# %%
"""
#### Box plots

We proceed with box plots 
"""


fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, sharey=True, figsize=(7.4, 4.), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.97, 'top': 0.89, 'bottom': 0.125,
                 'wspace': 0.06, 'width_ratios': [0.5, 1.5]})
fig2.suptitle("Figure 2: Violin plots of the sales vs categorical features",
              x=0.02, ha='left')


holiday_gdf = {i: df_['Weekly_Sales'] for i, df_ in df.groupby('Holiday_Flag')}
axs2[0].violinplot(list(holiday_gdf.values()), positions=list(holiday_gdf.keys()),
                   widths=0.5)

axs2[0].set_xlim(-0.5, 1.5)
axs2[0].set_xticks([0, 1], ['worked\n days', 'holidays'])
axs2[0].set_ylim(0, 3e6)
axs2[0].set_yticks(np.linspace(0, 3e6, 7), np.linspace(0, 3, 7))
axs2[0].set_ylabel('Weekly sales (M$)', labelpad=6)
axs2[0].grid(visible=True, linewidth=0.3)


store_gdf = {i: df_['Weekly_Sales'] for i, df_ in df.groupby('Store')}
axs2[1].violinplot(list(store_gdf.values()), positions=list(store_gdf.keys()),
                   widths=0.9)

axs2[1].set_xlim(0.3, 20.7)
axs2[1].set_xticks(np.arange(1, 21))
axs2[1].set_xlabel('Store number')
axs2[1].grid(visible=True, linewidth=0.3)

plt.show()

"""

For some stores, there are some events in which the weekly sales are about 10-20% higher
than the rest of the time.
This can be due to christmas / black friday
"""

# %%
"""
#### Scatter plots 

We conclude with scatter plots
"""

features = ['Date', 'Week', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']



fig3, axs3 = plt.subplots(
    nrows=3, ncols=2, sharey=True, figsize=(7, 8), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.96, 'top': 0.83, 'bottom': 0.065,
                 'wspace': 0.10, 'hspace': 0.3})
fig3.suptitle("Figure 3: Scatter plots of the weekly sales vs the various quantitative features",
              x=0.02, y=0.99, ha='left')


handles, labels = [], []
for istore, store_df in df.groupby('Store'):
    for feature, ax in zip(features, axs3.ravel()):
        line, = ax.plot(store_df[feature], store_df['Weekly_Sales'], color=COLORS[istore],
                         marker='o', markersize=4, linestyle='')
        # ax.plot(df[feature], df['Weekly_Sales'],
        #         marker='o', markersize=4, linestyle='')
        ax.set_xlabel(feature)
        ax.grid(visible=True, linewidth=0.3)
        ax.grid(visible=True, which='minor', linewidth=0.2)
    handles.append(line)
    labels.append(f'store {istore}')

for ax in axs3[:, 0]:
    ax.set_ylim(0, 3e6)
    ax.set_yticks(np.linspace(0, 3e6, 7), np.linspace(0, 3, 7))
    ax.set_ylabel('Weekly sales (M$)', labelpad=6)

dates = pd.date_range(start='1/1/2010', end='1/1/2013', freq='1YS')
axs3[0, 0].set_xticks(dates, dates.strftime('%Y'), rotation=0)
axs3[0, 0].set_xticks(pd.date_range(start='7/1/2010', end='7/1/2012', freq='12MS'),
                      minor=True)
axs3[0, 1].set_xlim(0, 53)
axs3[1, 0].set_xlim(10, 100)
axs3[2, 0].set_xlim(120, 230)
axs3[2, 1].set_xlim(4, 15)

fig3.legend(handles=handles, labels=labels,
            ncols=5, loc=(0.085, 0.84), alignment='center')

plt.show()

"""
We present in figure 3 scatter plots of the weekly sales for the various quantitative features.
There is no major dependence in any of the variables. However, we note the following:
- There is a significant increase in sales near the end of the year. This is probably an effect
of the christmas and holiday season. However, the relationship with the sales is clearly not linear.
- The sales tend to diminish as the CPI increases. Qualitatively, this means that customers spend less money
when they get less goods for the same amount.
- There is also possibly a small decrease of the sales as the temperature increases. It is not clear however
if it is a direct influence of temperature, or an indirect effect of the christmas season taking place
during a low-temperature period.
"""


# %% Data imputation

"""
## <a name="imputation"></a> Mssing data imputation

There are many missing data, we impute missing data with the following strategy.
We complete a missing entry with data from its corresponding store. In order:
1. Complete the `Date` by interpolating in decreasing order of preference, `CPI`, `Temperature`, `Unemployment`, `Fuel_Price`.
2. Complete `Year`, `Month`, `Week`
3. Complete `Temperature`, `Fuel_Price`, `CPI` and `Unemployment` by interpolating the values at neighboring dates
4. Complete the `Holiday_Flag` by the corresponding value if an entry with the same week number exists. Otherwise set it to `0`.

for that purpose, we use the raw dataframe, which contains the observations with missing target.
"""
# 1. complete the dates
gdf = raw_df.groupby('Store')
targets = ['CPI', 'Temperature', 'Unemployment', 'Fuel_Price']
miss_date = df.loc[df['Date'].isna()]

for i, row in miss_date.iterrows():
    for tgt in targets:
        if not row.isna()[tgt]:
            break
    xval = row[tgt]
    df_ = gdf.get_group(row['Store']).loc[:, ['Date', tgt]]
    dates = df_[~df_.isna().any(axis=1)]['Date'].astype('int64').values
    xvals = df_[~df_.isna().any(axis=1)][tgt].values
    y = np.interp(xval, xvals, dates)
    # impute missing value
    df.loc[i, 'Date'] = pd.Timestamp(y).round('7D') + pd.Timedelta('1D')

# 2. complete year, month, week
df.loc[:, 'Year'] = df['Date'].dt.year
df.loc[:, 'Month'] = df['Date'].dt.month
df.loc[:, 'Week'] = df['Date'].dt.isocalendar().week

# 3. complete the rest
for _, df_ in df.groupby('Store'):
    for tgt in targets:
        mask = df_[tgt].isna()
        if mask.any():
            idx = df_.loc[mask].index
            x_vals = df_.loc[~mask, 'Date'].astype('int64').values
            y_vals = df_.loc[~mask, tgt].values
            xs = df_.loc[mask, 'Date'].astype('int64').values
            df.loc[idx, tgt] = np.interp(xs, x_vals, y_vals)
    
# 4. complete holiday flag
for _, df_ in df.groupby('Store'):
    mask = df_['Holiday_Flag'].isna()
    idx = df_.loc[mask].index
    if idx.size > 0:
        holidays = np.zeros(len(idx), dtype=float)
        for i, week in enumerate(df_.loc[mask, 'Week']):
            fill_df = df_.loc[~mask, ['Week', 'Holiday_Flag']]
            week_match = (fill_df['Week'] == week)
            if week_match.any():
                # print(fill_df.loc[week_match])
                holidays[i] = fill_df.loc[week_match, 'Holiday_Flag'].iloc[0]
        df.loc[idx, 'Holiday_Flag'] = holidays

##
df.describe()


# %% Unregularized linear model






# %% Regularized linear model






# %% SVM


