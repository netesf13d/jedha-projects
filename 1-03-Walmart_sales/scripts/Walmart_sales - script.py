# -*- coding: utf-8 -*-
"""
Script version of the Walmart sales project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit, logit

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             root_mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     KFold,
                                     StratifiedKFold,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import LinearRegression, lasso_path, Lasso


# %% Loading
"""
## <a id="loading"></a> Data loading and preprocessing
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
"""
##
df = df.assign(Date=pd.to_datetime(df['Date'], format='%d-%m-%Y'))
df['Date'].dt.day_name()

# ##
df.loc[:, 'Year'] = df['Date'].dt.year
df.loc[:, 'Month'] = df['Date'].dt.month
df.loc[:, 'Week'] = df['Date'].dt.isocalendar().week


# %% EDA
"""
## <a id="eda"></a> Preliminary data analysis

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
## <a id="imputation"></a> Missing data imputation

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
"""
## <a id="linreg"></a> Unregularized linear model

Our data has 136 observations with about 30 features (`'Store'` is a categorical variable with 20 disctinct values).
With a ratio of less than 5 observations per feature, we are therefore at a high risk of overfitting. To circumvent
this problem, we can fit a very constrained model such as a linear regression.

In order to evaluate our model, we must split the data into train and test subsets. However, when doing so, we must be careful
to stratify according to the store index to ensure that all stores are represented in the train set. The risk would
be that a given store could be absent from the train set. The model would thus have of knowledge of the existence of
this store during training and crash when evaluating on the test set.

### Model construction and training
"""

## data preparation
y = df.loc[:, 'Weekly_Sales']
X = df.drop(['Weekly_Sales', 'Date'], axis=1) # Date is redundant with (Year, Month, Week)

# stratify according to the store index
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, stratify=X['Store'], random_state=1234)


## setup pipeline
# column preprocessing
cat_vars = ['Store']
bool_vars = ['Holiday_Flag']
quant_vars = ['Temperature','Fuel_Price', 'CPI',
              'Unemployment', 'Year', 'Month', 'Week']
col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop='first'), cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])

# full pipeline
linreg_model = Pipeline([('column_preprocessing', col_preproc),
                         ('regressor', LinearRegression())])

## fit and interpret
linreg_model.fit(X_tr, y_tr)


"""
### Evaluation of the model

We evaluate our model using the following metrics:
- The mean squared error (MSE), the standard evaluation of the prediction accuracy.
This is the value of the squared error loss at the model optimum.
- The root mean squared error (RMSE), the square root of the MSE, which measures the predicition accuracy.
- The coefficient of determination $R^2$, which represent the proportion of the target variance explained by the model.
- The mean absolute error (MAE), another measure of the prediction accuracy, less sensitive to outliers.
- The mean absolute percentage error (MAPE), which measures the relative prediction accuracy.
"""

## setup evaluation metrics
def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Helper function that evaluates the relevant evaluation metrics:
        MSE, RMSE, R-squared, MAE, MAPE
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = 100 * mean_absolute_percentage_error(y_true, y_pred)
    return mse, np.sqrt(mse), r2, mae, mape


## setup dataframe to hold the results
metric_names = ['MSE ($^2)', 'RMSE ($)', 'R2', 'MAE ($)', 'MAPE (%)']
index = pd.MultiIndex.from_product(
    [('linear regression', 'lasso regression'), ('train', 'test')],
    names=['model', 'eval. set'])
evaluation_df = pd.DataFrame(
    np.full((4, 5), np.nan), index=index, columns=metric_names)


## evaluate of train set
y_pred_tr = linreg_model.predict(X_tr)
evaluation_df.loc[('linear regression', 'train')] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = linreg_model.predict(X_test)
evaluation_df.loc[('linear regression', 'test')] = eval_metrics(y_test, y_pred_test)

##
evaluation_df.loc['linear regression']


"""
The performance of our simple model is quite good, with a typical prediction error of 100k$, or 10 %,
and a coefficient of determination close to 1. The evaluation metrics are similar for both train and test sets,
which show that there is no overfitting. This is remarkable for a model with 28 parameters trained on only 108 observations.
"""


"""
### Model interpretation

The interpretation of such a (scaled) linear model is simple: large coefficients correspond to more relevant features.
However, the interpretation must account for the fact that we dropped the first one-hot vector to remove
the redundancy in the categorical encoding.
"""

# recover feature names
col_preproc = linreg_model['column_preprocessing']
features = [feature_name.split('__')[1]
            for feature_name in col_preproc.get_feature_names_out()]

# get coefficients
intercept = linreg_model['regressor'].intercept_
coefs = linreg_model['regressor'].coef_


print(f'{"intercept":<12} : {intercept:>12.2f}')
for feature, coef in zip(features, coefs):
    print(f'{feature:<12} : {coef:>12.2f}')

"""
The intercept can be interpreted as the mean sales of store 1.
The coefficients `Store_<i>` correspond to the mean store sales relative to store 1.
We can see that these are the most relevant to the prediction of weekly sales.
The coefficients `Holidays`, `Temperature`, `Fuel_Price` and `Year` are smaller than the root mean squared error.
We can thus assume that these features have low relevance. The same is true for `Unemployment`,
which has a coefficient of the same order as the error. However, we note that its effect matches intuition:
a higher unemployment rate corresponds to a decrease of weekly sales. The consumer price index (`CPI`) is significant
with a positive coefficient. This has a simple economic interpretation. A higher CPI means that a given basket of products
has a higher price, which reflects in higher revenues for the store. Finally `Month` and `Week` both have large coefficients
of opposite sign. This is clearly a artifact: the model tries to account for the end of year sales increase with a linear combination
of `Month` and `Week`. This increase is a non-linear effect and we cannot expect the our linear regression will model it well.

To conclude, our model essentially tells that the sales are constant for each store, and tries to modelize at best the end-of-year increase.
This is already a very good modelization of our data (see figure 2).
"""


# %% Regularized linear model
"""
## <a id="ridge_linreg"></a> Regularized linear model

As we have seen, some features of the model are not very relevant. We may therefore introduce regularization
to get rid of potentially irrelevant variables. In this section we will do so with Lasso, or L1 regularization,
which has the advantage of enforcing coefficients sparsity.


### Model construction and parameter optimization

In order to get insights on the variable selection process of Lasso, we will compute a regularization path.
Cross validation is difficult here
Can't keep use gridsearch cause we want the params

"""

# No need to drop one-hot vector in column preprocessing
lasso_col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])
# Pipeline
lasso_model = Pipeline([('column_preprocessing', lasso_col_preproc),
                        ('regressor', Lasso(max_iter=1000000))])

## setup cross-validated grid search
reg = clone(lasso_model)
alphas = np.logspace(-1, 5, 301)
cv = KFold(n_splits=10, shuffle=True, random_state=1234)

## Run the search
metrics = np.zeros((5, len(alphas)))
coefs = np.zeros((28, len(alphas)))
for i, alpha in enumerate(alphas):
    if i% 10 == 0: print(i)
    reg['regressor'].alpha = alpha
    
    # construct validation predictions vector to compute validation metrics
    y_pred_v = np.zeros(len(y_tr), dtype=float)
    for itr, ival in cv.split(X_tr, y_tr):
        reg.fit(X_tr.iloc[itr], y_tr.iloc[itr])
        y_pred_v[ival] = reg.predict(X_tr.iloc[ival])
    metrics[:, i] = eval_metrics(y_tr, y_pred_v)
    
    # predict on full dataset to get regression coefficients
    reg.fit(X_tr, y_tr)
    coefs[:, i] = reg['regressor'].coef_
    # y_pred_v = reg.predict(X_tr)
    # metrics[:, i] = eval_metrics(y_tr, y_pred_v)


## define best regularization parameter as the argmin of MAPE (see below)
best_alpha = alphas[np.argmin(metrics[4])]


#%%
##
fig4, axs4 = plt.subplots(
    nrows=1, ncols=3, sharex=True, sharey=False, figsize=(9, 3.6), dpi=200,
    gridspec_kw={'left': 0.07, 'right': 0.98, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.28})
fig4.suptitle("Figure 4: Metrics vs the regularization parameter",
              x=0.02, ha='left')

# RMSE and MAE
axs4[0].plot(alphas, metrics[1], color='C0',
             label='Root mean sq. err.')
axs4[0].plot(alphas, metrics[3], color='C1',
             label='Mean abs. err')
axs4[0].axvline(best_alpha, color='k', linewidth=0.6)

axs4[0].set_xscale('log')
axs4[0].set_xlim(1e-1, 1e5)
axs4[0].set_xlabel('regularization parameter')
axs4[0].set_ylim(0, 8e5)
axs4[0].set_yticks(np.linspace(0, 8e5, 9),
                   [f'{i:.1f}' for i in np.linspace(0, 0.8, 9)])
axs4[0].set_ylabel('Prediction error (M$)')
axs4[0].grid(visible=True, linewidth=0.2)

axs4[0].legend(loc=(0.015, 0.81))

# R-squared
axs4[1].plot(alphas, metrics[2], color='C2')
axs4[1].axvline(best_alpha, color='k', linewidth=0.6)

axs4[1].set_xscale('log')
axs4[1].set_xlabel('regularization parameter')
axs4[1].set_ylim(0, 1)
axs4[1].set_ylabel('R-squared', labelpad=2)
axs4[1].grid(visible=True, linewidth=0.3)
axs4[1].grid(visible=True, which='minor', linewidth=0.2)

# MAPE
axs4[2].plot(alphas, metrics[4], color='C3')
axs4[2].axvline(best_alpha, color='k', linewidth=0.6)

axs4[2].set_xscale('log')
axs4[2].set_xlabel('regularization parameter')
axs4[2].set_ylim(0.7e0, 2e1)
axs4[2].set_ylabel('MAPE', labelpad=2)
axs4[2].grid(visible=True, linewidth=0.3)
axs4[2].grid(visible=True, which='minor', linewidth=0.2)


plt.show()


#%%
##
fig5, ax5 = plt.subplots(
    nrows=1, ncols=1, sharey=False, figsize=(6, 4.8), dpi=200,
    gridspec_kw={'left': 0.14, 'right': 0.96, 'top': 0.77, 'bottom': 0.11})
fig5.suptitle("Figure 5: Regression coefficients vs regularization parameter",
              x=0.02, ha='left')


for i in range(20):
    store_line, = ax5.plot(alphas, coefs[i], color='0.2', linewidth=0.3)

handles, labels = [store_line], ['Store_*'] # only one
for i in range(20, 28):
    line, = ax5.plot(alphas, coefs[i], color=COLORS[i-19])
    handles.append(line)
    labels.append(features[i-1])

ax5.axvline(best_alpha, color='k', linewidth=0.6)

ax5.set_xscale('log')
ax5.set_xlim(1e-1, 1e5)
ax5.set_xlabel('Regularization parameter')
ax5.set_ylim(-1e6, 1e6)
ax5.set_ylabel('Regression coefficient value')
ax5.grid(visible=True, linewidth=0.2)



fig5.legend(handles=handles, labels=labels,
            ncols=3, loc=(0.25, 0.79), alignment='center')

plt.show()


#%%
"""
### Evaluation of the model



"""

## train
lasso_model['regressor'].alpha = best_alpha
lasso_model.fit(X_tr, y_tr)



## evaluate of train set
y_pred_tr = lasso_model.predict(X_tr)
evaluation_df.loc[('lasso regression', 'train')] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = lasso_model.predict(X_test)
evaluation_df.loc[('lasso regression', 'test')] = eval_metrics(y_test, y_pred_test)

##
evaluation_df.loc['lasso regression']

"""
MAE/MAPE are the same as 

"""



# %% Conclusion

"""
## <a id="conclusion"></a> Conclusion

We conclude by comparing our results for both unregularized and regularized linear regressions.

"""


evaluation_df.loc['lasso regression']



