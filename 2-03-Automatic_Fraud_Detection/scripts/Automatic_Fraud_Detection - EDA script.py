# -*- coding: utf-8 -*-
"""
Script for the exploratory data analysis of the automatic fraud detection
project.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor


# %% Loading
"""
## <a id="loading"></a> Data loading and preprocessing



https://www.kaggle.com/datasets/kartik2112/fraud-detection/
"""

##
raw_df = pd.read_csv('../data/fraudTest.csv')
raw_df = raw_df.drop('Unnamed: 0', axis=1)

print(raw_df.describe(include='all'))

r"""
The dataset has 555715 rows, each corresponds to a transaction

We are in the presence of the following variables:
- Datetime-like variables
  - `trans_date_trans_time`, the transaction datetime
  - `dob`, the date of birth of the customer
  - `unix_time`, the unix time, probably of the transaction. For
  instance the value of the first row corresponds to 2013-06-21T12:14:25 UTC
  Which is the same as the corresponding `trans_date_trans_time`, save for the year
  (2013 vs 2020)
- Categorical variables
  - `cc_num`, the credit card number. It is a categorical variable,
  despite the fact that it is encoded as an integer.
  - `merchant`, the merchant name
  - `category`, the purchased goods category
  - `first`, `last`, `gender`, `job`, information about the customer
  - `street`, `city`, `state`, `zip`, location information about the merchant or customer
  - `trans_num`, the transaction number
- Boolean variables:
  - `is_fraud`, our target
- Quantitative variables:
  - `amt`, the transaction amount.
  - `city_pop`, the population of the city
  - `lat` and `long`, the geographic coordinates of the customer
  - `merch_lat` and `merch_long`, the geographic coordinates of the merchant

A lot of these variables are either irrelevant (eg the transaction number) or impossible
to use (eg the address). Moreover, some can be considered as personal data, and may not be
legal to use in a realistic context. We therefore produce a new set of features for fraud
detection:
- `month`, `weekday`, `day_time`, date and time of the transaction
- 'cos_day_time', 'sin_day_time'
- `amt`, the transaction amount
- 'cust_fraudster', indicates whether the customer has already commited fraud
- 'merch_fraud_victim', indicated whether the merchant has already been victim of fraud
"""

## cast the quantitative variable to float, for better handling of missing values
raw_df = raw_df.astype({'trans_date_trans_time': 'datetime64[s]'})

## discard the 'fraud_' string before each merchant value
raw_df['merchant'] = raw_df['merchant'].apply(lambda x: x.strip('fraud_'))

##
customers = [' '.join(cust)
             for cust in raw_df.loc[:, ['first', 'last']].to_numpy()]
customer_ids = dict(zip(set(customers), range(len(customers))))
raw_df = raw_df.assign(customer=customers)
customer_fraud = raw_df.loc[:, ['customer', 'is_fraud']].groupby('customer').any()
customer_fraud = dict(customer_fraud['is_fraud'].items())

## 
merchant_ids = dict(zip(set(raw_df['merchant']), range(len(raw_df))))
merchant_fraud = raw_df.loc[:, ['merchant', 'is_fraud']].groupby('merchant').any()
merchant_fraud = dict(merchant_fraud['is_fraud'].items())

##
dt = raw_df['trans_date_trans_time'].dt


##
df = pd.DataFrame({
    'merchant_id': raw_df['merchant'].map(merchant_ids),
    'customer_id': raw_df['customer'].map(customer_ids),
    'month': dt.month,
    'weekday': dt.weekday,
    'day_time': (raw_df['trans_date_trans_time'] - dt.floor('D')).dt.seconds,
    'amt': raw_df['amt'],
    'cust_fraudster': raw_df['customer'].map(customer_fraud),
    'merch_fraud_victim': raw_df['merchant'].map(merchant_fraud),
    'is_fraud': raw_df['is_fraud']
    })


# !!! make cos and sin time
# !!! cos and sin weekday

##
cat_vars = ['month', 'weekday']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['day_time', 'amt']

# %% EDA
"""
## <a id="eda"></a> Preliminary data analysis


"""
print(df.describe(include='all'))


"""
!!!
"""

# %%
"""
### Distribution of transfered amounts
"""

bins = np.linspace(0, 600, 151)
hist = np.histogram(df['amt'], bins=bins)


fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(6, 4), dpi=100,
    gridspec_kw={'left': 0.11, 'right': 0.94, 'top': 0.89, 'bottom': 0.14})
fig1.suptitle("Figure 1: Distribution of amounts spent in transactions",
              x=0.02, ha='left')

ax1.hist(df['amt'], bins=bins, density=False)

ax1.set_xlim(-0.5, 600)
ax1.set_xlabel('Amount spent')
ax1.set_ylim(0, 70000)
ax1.set_yticks(np.linspace(0, 70000, 8), np.arange(0, 80, 10))
ax1.set_ylabel('Counts (x1000)', labelpad=5)
ax1.grid(visible=True, axis='y', linewidth=0.3)
ax1.grid(visible=True, axis='y', which='minor', linewidth=0.2)


plt.show()

"""
!!!
Figure 1 presents 
"""

# %%
"""
### Correlation between non-categorical features
"""

## Correlation matrix
df_ = df.loc[:, ['is_fraud'] + quant_vars + bool_vars]
corr_df = df_.corr().to_numpy()

##
fig2, ax2 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.6, 4.6), dpi=100,
    gridspec_kw={'left': 0.15, 'right': 0.86, 'top': 0.72, 'bottom': 0.04, 'wspace': 0.24})
cax2 = fig2.add_axes((0.87, 0.04, 0.03, 0.68))
fig2.suptitle("Figure 2: Correlation matrix", x=0.02, ha='left')


ax2.set_aspect('equal')
cmap2 = plt.get_cmap('coolwarm').copy()
cmap2.set_extremes(under='0.9', over='0.5')
heatmap = ax2.pcolormesh(corr_df[::-1]*100, cmap=cmap2, vmin=-20, vmax=20,
                             edgecolors='0.2', linewidth=0.5)

ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ticklabels = ['Is fraud', 'Transaction\ntime', 'Transaction\namount',
              'Fraudster\ncustomer', 'Merchant\nvictim']
ax2.set_xticks(np.linspace(0.5, 4.5, 5), ticklabels, rotation=50,
               ha='center', fontsize=11)
ax2.set_yticks(np.linspace(4.5, 0.5, 5), ticklabels, rotation=0,
               ha='right', fontsize=11)

for (i, j), c in np.ndenumerate(corr_df[::-1]*100):
    ax2.text(j+0.52, i+0.45, f'{c:.1f}', ha='center', va='center', fontsize=9)

pos = ax2.get_position().bounds
x, y = pos[0], pos[1] + pos[3]

fig2.colorbar(heatmap, cax=cax2, orientation="vertical", ticklocation="right")
cax2.text(1.4, 1.03, 'Correlation\ncoef. (%)', fontsize=11, ha='center',
          transform=cax2.transAxes)


plt.show()

"""
Figure 2 presents
!!!
"""

# %%
"""
### Fraud probability vs time

"""

## Fraud probability vs month
df_ = df[['month', 'is_fraud']].assign(count=1).groupby('month').sum()

months = df_.index.to_numpy()
is_fraud = df_['is_fraud'].to_numpy()
count = df_['count'].to_numpy()

month_fprob = is_fraud / count
month_fprob_err = month_fprob * (1 - month_fprob) / np.sqrt(count)


## Fraud probability vs week day
df_ = df[['weekday', 'is_fraud']].assign(count=1).groupby('weekday').sum()

weekdays = df_.index.to_numpy()
is_fraud = df_['is_fraud'].to_numpy()
count = df_['count'].to_numpy()

day_fprob = is_fraud / count
day_fprob_err = day_fprob * (1 - day_fprob) / np.sqrt(count)


## Fraud probability vs hour
df_ = df[['day_time', 'is_fraud']]
df_ = df_.assign(count=1, hour=df_['day_time'] // 3600)
df_ = df_.groupby('hour').sum()

hours = df_.index.to_numpy()
is_fraud = df_['is_fraud'].to_numpy()
count = df_['count'].to_numpy()

hour_fprob = is_fraud / count
hour_fprob_err = hour_fprob * (1 - hour_fprob) / np.sqrt(count)



##
fig3, axs3 = plt.subplots(
    nrows=1, ncols=3, figsize=(10, 3.8), dpi=100, sharey=True,
    gridspec_kw={'left': 0.065, 'right': 0.95, 'top': 0.87, 'bottom': 0.16,
                 'wspace': 0.06})
fig3.suptitle("Figure 3: Time-dependence of the fraud probability",
              x=0.02, y=0.97, ha='left')

axs3[0].errorbar(months, 100*month_fprob, yerr=100*month_fprob_err,
                 color='tab:orange', linestyle='', marker='o', markersize=4)
axs3[0].tick_params(which='both', direction='in', top=True, right=True)
axs3[0].set_xlim(0, 13)
axs3[0].set_xticks(np.arange(1, 13))
axs3[0].set_xlabel('Month')
axs3[0].set_ylim(0, 2.)
axs3[0].set_yticks(np.linspace(0, 2, 5))
axs3[0].set_yticks(np.linspace(0.25, 1.75, 4), minor=True)
axs3[0].grid(visible=True, linewidth=0.3)
axs3[0].grid(visible=True, which='minor', linewidth=0.2)


axs3[1].errorbar(weekdays, 100*day_fprob, yerr=100*day_fprob_err,
                 color='tab:orange', linestyle='', marker='o', markersize=4)
axs3[1].tick_params(which='both', direction='in', top=True, right=True)
axs3[1].set_xlim(-0.5, 6.5)
axs3[1].set_xticks(np.arange(7),
                   ['Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.'])
axs3[1].set_xlabel('Week day', labelpad=8)
axs3[1].grid(visible=True, linewidth=0.3)
axs3[1].grid(visible=True, which='minor', linewidth=0.2)


axs3[2].errorbar(hours, 100*hour_fprob, yerr=100*hour_fprob_err,
                 color='tab:orange', linestyle='', marker='o', markersize=4)
axs3[2].tick_params(which='both', direction='in', top=True,
                    right=True, labelright=True)
axs3[2].set_xlim(-1.5, 24.5)
axs3[2].set_xticks(np.arange(0, 24, 2))
axs3[2].set_xticks(np.arange(1, 24, 2), minor=True)
axs3[2].set_xlabel('Hour')
axs3[2].grid(visible=True, linewidth=0.3)
axs3[2].grid(visible=True, which='minor', linewidth=0.2)


fig3.text(0.02, 0.51, 'Fraud probability (%)', rotation=90, fontsize=11,
          ha='center', va='center')


plt.show()

"""
Figure 3 presents
!!!

Strong non-linearity in the time dependence of the fraud -> choose non-linear model
such as decision tree.
"""


# %%
"""
### Joint probability distributions

!!!
"""

## Transaction amounts and 1D histogram bins
amount = df['amt'].to_numpy()
amt_bins = np.linspace(0, 1200, 121)

## Day time observations and 1D histogram bins
day_time = df['day_time'].to_numpy()
time_bins = np.linspace(0, 86400, 49)

## distrib
yx = np.meshgrid(time_bins, amt_bins, indexing='ij')
time_amt_hist = np.histogram2d(day_time, amount, bins=[time_bins, amt_bins])[0]

## fraud probability
is_fraud = df['is_fraud'].to_numpy()
amt_j = np.digitize(amount, amt_bins) - 1
time_i = np.digitize(day_time, time_bins) - 1
fraud_prob = np.zeros((len(time_bins)-1, len(amt_bins)-1))
for i in range(len(time_bins)-1):
    idx_i = (time_i == i)
    for j in range(len(amt_bins)-1):
        fraud_prob[i, j] = np.mean(is_fraud[(amt_j == j) * idx_i])
# for i, j in np.ndindex(fraud_prob.shape):


# %%
##
fig4, axs4 = plt.subplots(
    nrows=1, ncols=2, figsize=(8.8, 4.4), dpi=100,
    gridspec_kw={'left': 0.07, 'right': 0.89, 'top': 0.87, 'bottom': 0.11,
                 'wspace': 0.34})
fig4.suptitle("Figure 4: Joint probability distribution of rental price with quantitative variables",
              x=0.02, ha='left')



cmap4_0 = plt.get_cmap('hot').copy()
cmap4_0.set_extremes(bad='0.6', under=(0.0416, 0.0, 0.0, 1.0), over='0.6')
heatmap = axs4[0].pcolormesh(*yx, np.log(time_amt_hist),
                             cmap=cmap4_0, vmin=1, vmax=10)

axs4[0].set_xlim(0, np.max(time_bins))
axs4[0].set_xticks(np.linspace(0, 86400, 13), np.arange(0, 25, 2))
axs4[0].set_xlabel('Hour', fontsize=11)
axs4[0].set_ylim(0, np.max(amt_bins))
axs4[0].set_yticks(np.arange(50, 500, 100), minor=True)
axs4[0].set_ylabel('Rental price', fontsize=11)
axs4[0].set_title('Mileage x Rental price')

cax4_0 = fig4.add_axes((0.415, 0.12, 0.025, 0.76))
fig4.colorbar(heatmap, cax=cax4_0, orientation="vertical", ticklocation="right")
cax4_0.set_yticks(np.linspace(1, 10, 10))
cax4_0.text(1.1, -0.09, 'Counts', fontsize=11, ha='center',
            transform=cax4_0.transAxes)


cmap4_1 = plt.get_cmap('hot').copy()
cmap4_1.set_extremes(bad='0.6', under='0.7', over='0.5')
heatmap = axs4[1].pcolormesh(*yx, fraud_prob,
                             cmap=cmap4_1, vmin=0, vmax=1.1)
axs4[1].set_xlim(0, 86400)
axs4[1].set_xticks(np.linspace(0, 86400, 13), np.arange(0, 25, 2))
axs4[1].set_xlabel('Hour', fontsize=11)
axs4[0].set_ylim(0, np.max(amt_bins))

cax4_1 = fig4.add_axes((0.915, 0.12, 0.025, 0.76))
fig4.colorbar(heatmap, cax=cax4_1, orientation="vertical", ticklocation="right")
cax4_1.set_yticks(np.linspace(0, 1, 6))
cax4_1.text(1.1, -0.09, 'Counts', fontsize=11, ha='center',
            transform=cax4_1.transAxes)

axs4[1].set_title('Fraud probability')



plt.show()

"""
Figure 4 presents
!!!
"""


# %% Preprocessing and utilities
r"""
## <a id="preproc_utils"></a> Data preprocessing and utilities

!!!
Before moving on to model construction and training, we must first setup dataset
preprocessing and preparation.
We also introduce here some utilities relevant to model evaluation carried later on.


### Model evaluation utilities

We evaluate our model using the following metrics:

"""

def eval_metrics(cm: np.ndarray,
                  print_cm: bool = True,
                  print_precrec: bool = True)-> None:
    """
    !!!
    Print metrics related to the confusion matrix: precision, recall, F1-score.
    """
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1.
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1.
    if print_cm:
        print("Confusion matrix\n", cm / np.sum(cm))
    if print_precrec:
        print(f'Precision: {prec:.8}; recall: {recall:.8}')
    print(f'F1-score: {2*prec*recall/(prec+recall):.8}')


"""

### Data preprocessing

The preprocessing of the dataset consists in the removal of outliers. We choose to remove those
observations corresponding to high rental price and mileage using a quantile range criterion.
More precisely, we discard those observations $i$ such that
$x_i < m + k(Q_{0.75} - Q_{0.25})$, where
$Q_{\alpha}$ is the $\alpha$-quantile and $k=6$ is an exclusion factor. We filter only
the values above, since the quantities are positive by definition.

We also setup here the preprocessing pipeline that will be part of the pricing models:
one-hot encoding of categorical variables and scaling of quantitative variables.
"""





## data preparation
y = df['is_fraud']
X = df.drop(['is_fraud', 'merchant_id', 'customer_id'], axis=1)

## train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)












# %%



## Remove outliers
q1, q2 = 0.25, 0.75
k = 6 # exclusion factor
quantiles = df.loc[:, ['mileage', 'rental_price_per_day']].quantile([q1, q2])
iqr_mil, iqr_pr = quantiles.iloc[1] - quantiles.iloc[0]
med_mil, med_pr = df.loc[:, ['mileage', 'rental_price_per_day']].median()

loc = ((df['mileage'] < med_mil + k*iqr_mil)
       * (df['rental_price_per_day'] < med_pr + k*iqr_pr))
processed_df = df.loc[loc]

processed_df.describe(include='all')

## data preparation
y = processed_df['rental_price_per_day']
X = processed_df.drop('rental_price_per_day', axis=1)



## column preprocessing
cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']
bool_vars = ['private_parking_available', 'has_gps', 'has_air_conditioning',
             'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
             'winter_tires']
quant_vars = ['mileage', 'engine_power']
col_preproc = ColumnTransformer(
    [('cat_ohe',
      OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.005),
      cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])


"""
### Model evaluation utilities

We evaluate our model using the following metrics:
- The mean squared error (MSE), the standard evaluation of the prediction accuracy.
- The root mean squared error (RMSE), the square root of the MSE, which measures the predicition accuracy.
- The coefficient of determination $R^2$, which represent the proportion of the target variance explained by the model.
- The mean absolute error (MAE), another measure of the prediction accuracy, less sensitive to outliers.
- The mean absolute percentage error (MAPE), which measures the relative prediction accuracy.
"""

## Evaluation metrics helper function
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


## Dataframe to hold the results
metric_names = ['MSE', 'RMSE', 'R2', 'MAE', 'MAPE (%)']
index = pd.MultiIndex.from_product(
    [('Ridge', 'GBM'), ('train', 'test')],
    names=['model', 'eval. set'])
evaluation_df = pd.DataFrame(
    np.full((4, 5), np.nan), index=index, columns=metric_names)


# %% Ridge
r"""
## <a id="linreg"></a> Ridge regression

We first begin by training a baseline ridge regression model.


### Parameters optimization

We introduce L2 regularization to the model because of the large number of categorical features.
In order to select the appropriate value for the regularization parameter $\alpha$, we perform a grid search
with cross validation.
"""

## column preprocessing
cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']
bool_vars = ['private_parking_available', 'has_gps', 'has_air_conditioning',
             'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
             'winter_tires']
quant_vars = ['mileage', 'engine_power']
col_preproc = ColumnTransformer(
    [('cat_ohe',
      OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.005),
      cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])




## Grid search of the regularization parameter with cross validation
scoring = ('neg_mean_squared_error',  'neg_root_mean_squared_error', 'r2',
           'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error')
alphas = np.logspace(-1, 5, 61)
reg = Pipeline(
    [('column_preprocessing', col_preproc),
     ('regressor', GridSearchCV(Ridge(), param_grid={'alpha': alphas},
                                scoring=scoring, refit=False, cv=10))]
)

reg.fit(X_tr, y_tr)


## Extract the relevant metrics
cv_res = reg['regressor'].cv_results_
rmse = -cv_res['mean_test_neg_root_mean_squared_error']
r2 = cv_res['mean_test_r2']
mae = -cv_res['mean_test_neg_mean_absolute_error']
mape = -cv_res['mean_test_neg_mean_absolute_percentage_error']

#%%
##
fig5, axs5 = plt.subplots(
    nrows=1, ncols=3, sharex=True, sharey=False, figsize=(8.8, 3.6), dpi=120,
    gridspec_kw={'left': 0.06, 'right': 0.98, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.22})
fig5.suptitle("Figure 5: Scoring metrics vs ridge regularization parameter",
              x=0.02, ha='left')

# RMSE and MAE
axs5[0].plot(alphas, rmse, color='C0', label='Root mean sq. err.')
axs5[0].plot(alphas, mae, color='C1', label='Mean abs. err')
axs5[0].set_xscale('log')
axs5[0].set_xlim(1, 1e5)
axs5[0].set_xticks(np.logspace(0, 5, 6))
axs5[0].set_xticks(
    np.concatenate([np.linspace(10**i, 10**(i+1), 10) for i in range(-1, 5)]),
    minor=True)
axs5[0].set_ylim(10, 35)
axs5[0].set_yticks(np.linspace(12.5, 32.5, 5), minor=True)
axs5[0].set_ylabel('Price error')
axs5[0].grid(visible=True, linewidth=0.3)
axs5[0].grid(visible=True, which='minor', linewidth=0.2)
axs5[0].legend(loc=(0.015, 0.81))

# R-squared
axs5[1].plot(alphas, r2, color='C2', label='R-squared')
axs5[1].set_xlabel(r'Regularization parameter $\alpha$', fontsize=11)
axs5[1].set_ylim(0, 1)
axs5[1].grid(visible=True, linewidth=0.3)
axs5[1].grid(visible=True, which='minor', linewidth=0.2)
axs5[1].legend(loc=(0.015, 0.89))

# MAPE
axs5[2].plot(alphas, 100*mape, color='C3', label='MAPE')
axs5[2].set_ylim(12, 26)
axs5[2].grid(visible=True, linewidth=0.3)
axs5[2].grid(visible=True, which='minor', linewidth=0.2)
axs5[2].legend(loc=(0.015, 0.89))


plt.show()

r"""
Figure 5 presents plots of our selected metrics as a function of
the regularization parameter $\alpha$ of the ridge regression.
The performance of the model starts to degrade for values of $\alpha$ above $\sim 100$.
"""


# %%
r"""
### Model training and evaluation

According to the previous results, we retain a value $\alpha = 100$ for
the regularization parameter of our ridge model.
"""

## Complete pipeline
ridge_model = Pipeline([('column_preprocessing', col_preproc),
                        ('regressor', Ridge(alpha=100))])

## train
ridge_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = ridge_model.predict(X_tr)
evaluation_df.iloc[0] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = ridge_model.predict(X_test)
evaluation_df.iloc[1] = eval_metrics(y_test, y_pred_test)


print(evaluation_df.loc['Ridge'])

r"""
The performance of the model is acceptable, with $R^2 \simeq 0.7$.
The mean absolute error is about 12-13, significantly lower than the root mean squared error (17-19).
This indicates that our model deals poorly with outliers. This can be caused by the aggregation of
infrequent categories that may correspond large price differences (especially the car brand).

The metrics are significantly better for the train set than for the test set (300 vs 370 mean squared error),
indicating an important overfitting of the model. Another possible explanation could be that the test set
contains many observations with values in the infrequent categories, on which the model will fail to
produce good results.
"""

#%%
"""
### Model interpretation

We inspect the model coefficients to determine which variables are relevant to rental price.
"""

## Recover feature names
col_preproc_ = ridge_model['column_preprocessing']
features = ['intercept']
features += [feature_name.split('__')[1].split('_sklearn')[0]
             for feature_name in col_preproc_.get_feature_names_out()]

## Get coefficients
intercept = ridge_model['regressor'].intercept_
coef_vals = np.concatenate([[intercept], ridge_model['regressor'].coef_])


for feature, coef in zip(features, coef_vals):
    print(f'{feature:<26} : {coef:>6.2f}')

"""
The two most important factors influencing the rental price are the mileage and engine power.
This is in agreement with what was observed on the correlation matrix (see figure 2).
The presence of car options, especially GPS, is also important to the pricing.
Within a given categorical feature, some categories appear have an important offset with the average price (intercept)
while other do not. Compare for instance `car_type_suv` and `car_type_sedan`.
This could be an artefact of the strong correlation between the prediction features.
"""


# %% Gradient Boosting model
"""
## <a id="gbm"></a> Gradient Boosting model

We complete the pricing models catalogue of our application by training
a gradient boosting regressor.


### Parameters optimization
"""

## Grid search of the regularization parameter with cross validation
scoring = ('neg_mean_squared_error',  'neg_root_mean_squared_error', 'r2',
           'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error')
n_estimators_vals = np.arange(100, 1001, 100)
reg = Pipeline(
    [('column_preprocessing', col_preproc),
     ('regressor', GridSearchCV(GradientBoostingRegressor(random_state=1234),
                                param_grid={'n_estimators': n_estimators_vals},
                                scoring=scoring, refit=False, cv=5))]
)

reg.fit(X_tr, y_tr)

#%%

## Extract the relevant metrics
cv_res = reg['regressor'].cv_results_
rmse = -cv_res['mean_test_neg_root_mean_squared_error']
r2 = cv_res['mean_test_r2']
mae = -cv_res['mean_test_neg_mean_absolute_error']
mape = -cv_res['mean_test_neg_mean_absolute_percentage_error']


##
fig6, axs6 = plt.subplots(
    nrows=1, ncols=3, sharex=True, sharey=False, figsize=(9, 3.6), dpi=120,
    gridspec_kw={'left': 0.06, 'right': 0.98, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.24})
fig6.suptitle("Figure 6: Scoring metrics vs the number of gradient boosting estimators",
              x=0.02, ha='left')

# RMSE and MAE
axs6[0].plot(n_estimators_vals, rmse, color='C0', label='Root mean sq. err.')
axs6[0].plot(n_estimators_vals, mae, color='C1', label='Mean abs. err')
axs6[0].set_xlim(0, 1000)
axs6[0].set_ylim(10, 18)
axs6[0].set_yticks(np.linspace(10.5, 17.5, 8), minor=True)
axs6[0].set_ylabel('Price error')
axs6[0].grid(visible=True, linewidth=0.3)
axs6[0].grid(visible=True, which='minor', linewidth=0.2)
axs6[0].legend(loc=(0.015, 0.81))

# R-squared
axs6[1].plot(n_estimators_vals, r2, color='C2', label='R-squared')
axs6[1].set_xlabel(r'Number of estimators `n_estimators`', fontsize=11)
axs6[1].set_ylim(0.7, 0.8)
axs6[1].grid(visible=True, linewidth=0.3)
axs6[1].grid(visible=True, which='minor', linewidth=0.2)
axs6[1].legend(loc=(0.015, 0.89))

# MAPE
axs6[2].plot(n_estimators_vals, 100*mape, color='C3', label='MAPE')
axs6[2].set_ylim(12, 13)
axs6[2].grid(visible=True, linewidth=0.3)
axs6[2].grid(visible=True, which='minor', linewidth=0.2)
axs6[2].legend(loc=(0.61, 0.89))


plt.show()

r"""
Figure 6 presents plots of our selected metrics as a function of the number of estimators
used with the gradient boosting regressor. The metrics reach an optimum for `n_estimators = 500`,
which we retain for the model.
"""


# %%
"""
### Model training and evaluation
"""

## Complete pipeline
gb_model = Pipeline(
    [('column_preprocessing', col_preproc),
     ('regressor', GradientBoostingRegressor(n_estimators=500, random_state=1234))]
)

## train
gb_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = gb_model.predict(X_tr)
evaluation_df.iloc[2] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = gb_model.predict(X_test)
evaluation_df.iloc[3] = eval_metrics(y_test, y_pred_test)

print(evaluation_df)

"""
The gradient boosting model is more performant than the simple ridge model.
However, overfitting is still present, and even worse than for ridge regression.
Nevertheless, the test set metrics are quite close
to those obtained by cross-validation (see figure 6).
"""