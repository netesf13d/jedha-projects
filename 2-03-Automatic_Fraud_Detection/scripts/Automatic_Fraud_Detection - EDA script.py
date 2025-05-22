# -*- coding: utf-8 -*-
"""
Script for the exploratory data analysis of the automatic fraud detection
project.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)


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
- `amt`, the transaction amount
- `state`, the merchant or customer state
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
    'state': raw_df['state'],
    'cust_fraudster': raw_df['customer'].map(customer_fraud),
    'merch_fraud_victim': raw_df['merchant'].map(merchant_fraud),
    'is_fraud': raw_df['is_fraud']
    })

r"""
!!!
In addition, since the day time and week day are cyclic (e.g., 1h is as close to 23h
as 15h is close to 13h), we create derived features by taking the sine and cosine of
these variables.

$$\cos \left( 2 k \pi X / T(X) \right), \quad \cos \left( 2 k \pi X / T(X) \right)$$


To keep thins simple, we limit ourselves to $k=1$ to capture simple periodic structures.
This gives, for instance, in the case of `day_time`, the two features
$$\cos(2 \pi \mathrm{day_time} / \mathrm{24h}), \quad \cos(2 \pi \mathrm{day_time} / \mathrm{24h})$$
"""

## Add cosine and sine of periodic features
df = df.assign(cos_day_time=np.cos(2*np.pi*df['day_time']/86400),
               sin_day_time=np.sin(2*np.pi*df['day_time']/86400),)
               # cos_weekday=np.cos(2*np.pi*df['weekday']/7),
               # sin_weekday=np.sin(2*np.pi*df['weekday']/7))

##
cat_vars = ['month', 'weekday', 'state']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['day_time', 'amt', 'cos_day_time', 'sin_day_time']

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

##
amt_vals = np.logspace([0], [5], 101)
amt_icdf = np.sum(df['amt'].to_numpy() > amt_vals, axis=1) / len(df['amt'])

##
fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(6, 4), dpi=100,
    gridspec_kw={'left': 0.11, 'right': 0.94, 'top': 0.89, 'bottom': 0.14})
fig1.suptitle("Figure 1: Distribution of amounts spent in transactions",
              x=0.02, ha='left')

ax1.plot(amt_vals, amt_icdf, linestyle='', marker='+', markersize=5)

ax1.set_xscale('log')
ax1.set_xlim(1, 1e5)
ax1.set_xlabel('Amount spent')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 1)
ax1.set_ylabel('Counts (x1000)', labelpad=5)
ax1.grid(visible=True, linewidth=0.3)
ax1.grid(visible=True, which='minor', linewidth=0.2)


plt.show()

"""
!!!
Figure 1 presents the distribution of transaction amounts. Most values are below
200.
"""

# %%
"""
### Correlation between non-categorical features
"""

## Correlation matrix
df_ = df.loc[:, ['is_fraud'] + quant_vars + bool_vars]
corr = df_.corr().to_numpy()

##
fig2, ax2 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.6, 4.6), dpi=100,
    gridspec_kw={'left': 0.15, 'right': 0.86, 'top': 0.72, 'bottom': 0.04, 'wspace': 0.24})
cax2 = fig2.add_axes((0.87, 0.04, 0.03, 0.68))
fig2.suptitle("Figure 2: Correlation between non-categorical features",
              x=0.02, ha='left')


ax2.set_aspect('equal')
cmap2 = plt.get_cmap('bwr').copy()
cmap2.set_extremes(under='0.4', over='0.4')
heatmap = ax2.pcolormesh(corr[::-1]*100, cmap=cmap2, vmin=-20, vmax=20,
                             edgecolors='0.2', linewidth=0.5)

ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ticklabels = ['Is fraud', 'Transaction\ntime', 'Transaction\namount',
              'cosine\ntrans. time', 'sine\ntrans. time',
              'Fraudster\ncustomer', 'Merchant\nvictim']
ax2.set_xticks(np.linspace(0.5, 6.5, 7), ticklabels, rotation=50,
               ha='center', fontsize=11)
ax2.set_yticks(np.linspace(6.5, 0.5, 7), ticklabels, rotation=0,
               ha='right', fontsize=11)

for (i, j), c in np.ndenumerate(corr[::-1]*100):
    ax2.text(j+0.52, i+0.45, f'{c:.1f}', color='w' if abs(c) > 50 else 'k',
             ha='center', va='center', fontsize=9)

pos = ax2.get_position().bounds
x, y = pos[0], pos[1] + pos[3]

fig2.colorbar(heatmap, cax=cax2, orientation="vertical", ticklocation="right")
cax2.text(1.4, 1.07, 'Correlation\ncoef. (%)', fontsize=11, ha='center',
          transform=cax2.transAxes)


plt.show()

"""
Figure 2 shows the correlation matrix of non-categorical variables.
The linear correlation between fraud events and other variables is rather weak.
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
amt_bins = np.linspace(0, 1800, 91)

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
    nrows=1, ncols=2, figsize=(8.8, 4.4), dpi=100, sharex=True,
    gridspec_kw={'left': 0.07, 'right': 0.91, 'top': 0.87, 'bottom': 0.11,
                 'wspace': 0.48})
fig4.suptitle("Figure 4: Joint probability distribution of rental price with quantitative variables",
              x=0.02, ha='left')

cmap4_0 = plt.get_cmap('hot').copy()
cmap4_0.set_extremes(bad='0.6', under=(0.0416, 0.0, 0.0, 1.0), over='0.6')
heatmap = axs4[0].pcolormesh(*yx, np.log(time_amt_hist),
                             cmap=cmap4_0, vmin=1, vmax=9)

axs4[0].set_xlim(0, np.max(time_bins))
axs4[0].set_xticks(np.linspace(0, 86400, 13), np.arange(0, 25, 2))
axs4[0].set_xlabel('Hour', fontsize=11)
axs4[0].set_ylim(0, np.max(amt_bins))
axs4[0].set_yticks(np.arange(0, 1900, 200),
                   [f'{x:.1f}' for x in np.linspace(0, 1.8, 10)])
axs4[0].set_ylabel('Transaction amount (x1000)', fontsize=11)
axs4[0].set_title('Time x transaction amount')

cax4_0 = fig4.add_axes((axs4[0]._position.get_points()[1, 0]+0.02,
                        axs4[0]._position.y0,
                        0.02,
                        axs4[0]._position.height))
fig4.colorbar(heatmap, cax=cax4_0, orientation="vertical", ticklocation="right")
cax4_0.set_yticks(np.linspace(1, 9, 9),
                  [rf'$10^{x}$' for x in np.arange(1, 10)])
cax4_0.text(1.1, -0.07, 'Counts', fontsize=11, ha='center', va='center',
            transform=cax4_0.transAxes)


cmap4_1 = plt.get_cmap('hot').copy()
cmap4_1.set_extremes(bad='0.6', under='0.7', over='0.5')
heatmap = axs4[1].pcolormesh(*yx, fraud_prob,
                             cmap=cmap4_1, vmin=0, vmax=1.1)
axs4[1].set_xlim(0, 86400)
axs4[1].set_xticks(np.linspace(0, 86400, 13), np.arange(0, 25, 2))
axs4[1].set_xlabel('Hour', fontsize=11)
axs4[1].set_yticks(np.arange(0, 1900, 200),
                   [f'{x:.1f}' for x in np.linspace(0, 1.8, 10)])
axs4[1].set_ylabel('Transaction amount (x1000)', fontsize=11)

cax4_1 = fig4.add_axes((axs4[1]._position.get_points()[1, 0]+0.02,
                        axs4[1]._position.y0,
                        0.02,
                        axs4[1]._position.height))
fig4.colorbar(heatmap, cax=cax4_1, orientation="vertical", ticklocation="right")
cax4_1.set_ylim(0, 1)
cax4_1.set_yticks(np.linspace(0, 1, 6))
cax4_1.text(1.1, -0.12, 'Fraud\n prob.', fontsize=11, ha='center',
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
Before moving on to model constructicat_vars = ['month', 'weekday']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['day_time', 'amt', 'cos_day_time', 'sin_day_time']on and training, we must first setup dataset
preprocessing and preparation.
We also introduce here some utilities relevant to model evaluation carried later on.


### Model evaluation utilities

We evaluate our model using the following metrics:
- The confusion matrix
- The precision
- The recall
- The F1 score

"""

def eval_metrics(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 print_cm: bool = False,
                 print_f1: bool = False)-> None:
    """
    Helper function that evaluates and return the relevant evaluation metrics:
        confusion matrix, precision, recall, F1-score
    """
    cm = confusion_matrix(y_true, y_pred)
    
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1.
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1.
    f1 = 2*prec*recall/(prec+recall)
    
    if print_cm:
        print("Confusion matrix\n", cm / np.sum(cm))
    if print_f1:
        print(f'Precision: {prec:.8}; recall: {recall:.8}')
        print(f'F1-score: {2*prec*recall/(prec+recall):.8}')
    
    return cm, prec, recall, f1


## Dataframe to hold the results
metric_names = ['precision', 'recall', 'F1-score']
index = pd.MultiIndex.from_product(
    [('Log reg', 'RF', 'HGB'), ('train', 'test')],
    names=['model', 'eval. set'])
evaluation_df = pd.DataFrame(
    np.full((6, 3), np.nan), index=index, columns=metric_names)

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
    X, y, test_size=0.2, stratify=y, random_state=1234)


# %% Logistic regression
r"""
## <a id="logreg"></a> Logistic regression

We first begin by training a baseline ridge regression model.


### Parameters optimization

We introduce L2 regularization to the model because of the large number of categorical features.
In order to select the appropriate value for the regularization parameter $\alpha$, we perform a grid search
with cross validation.

!!!
"""

## column preprocessing
cat_vars = ['month', 'weekday', 'state']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['amt', 'cos_day_time', 'sin_day_time']

col_preproc = ColumnTransformer(
    [('cat_ohe',
      OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.005),
      cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])



## Grid search of the regularization parameter with cross validation
scoring = ('precision',  'recall', 'f1')
Cs = np.logspace(-1, 4, 11)
logreg = LogisticRegression(penalty='l2')
clf = Pipeline(
    [('column_preprocessing', col_preproc),
     ('classifier', GridSearchCV(logreg, param_grid={'C': Cs},
                                 scoring=scoring, refit=False, cv=5))]
)
clf.fit(X_tr, y_tr)


## Extract the relevant metrics
cv_res = clf['classifier'].cv_results_
prec = cv_res['mean_test_precision']
std_prec = cv_res['std_test_precision']
recall = cv_res['mean_test_recall']
std_recall = cv_res['std_test_recall']
f1 = cv_res['mean_test_f1']
std_f1 = cv_res['std_test_f1']


# %%

##
fig5, ax5 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6, 4), dpi=120,
    gridspec_kw={'left': 0.11, 'right': 0.95, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.22})
fig5.suptitle("Figure 5: Scoring metrics vs ridge regularization parameter",
              x=0.02, ha='left')

# RMSE and MAE
ax5.plot(Cs, prec, color='C0', label='Precision')
ax5.fill_between(Cs, prec-std_prec, prec+std_prec,
                 color='C0', alpha=0.2)
ax5.plot(Cs, recall, color='C1', label='Recall')
ax5.fill_between(Cs, recall-std_recall, recall+std_recall,
                 color='C1', alpha=0.2)
ax5.plot(Cs, f1, color='C2', label='F1-score')
ax5.fill_between(Cs, f1-std_f1, f1+std_f1,
                 color='C2', alpha=0.2)


ax5.set_xscale('log')
ax5.set_xlim(1e-1, 1e4)
ax5.set_xticks(np.logspace(-1, 4, 6))
ax5.set_xticks(
    np.concatenate([np.linspace(10**i, 10**(i+1), 10) for i in range(-1, 4)]),
    minor=True)
ax5.set_xlabel('Regularization parameter (C)')
ax5.set_ylim(0, 1)
ax5.set_ylabel('Price error')
ax5.grid(visible=True, linewidth=0.3)
ax5.grid(visible=True, which='minor', linewidth=0.2)
ax5.legend(loc=(0.015, 0.9), ncols=3)


plt.show()

r"""
Figure 5 presents
!!!
"""

# %%
r"""
### Model training and evaluation

According to the previous results, we retain a value $\alpha = 100$ for
the regularization parameter of our ridge model.
"""

## Complete pipeline
logreg_model = Pipeline([('column_preprocessing', col_preproc),
                        ('classifier', LogisticRegression(C=100))])

## train
logreg_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = logreg_model.predict(X_tr)
_, prec, recall, f1 = eval_metrics(y_tr, y_pred_tr)
evaluation_df.iloc[0] = prec, recall, f1

## evaluate on test set
y_pred_test = logreg_model.predict(X_test)
_, prec, recall, f1 = eval_metrics(y_test, y_pred_test)
evaluation_df.iloc[1] = prec, recall, f1

print(evaluation_df.loc['Log reg'])

r"""
!!!
"""

# %% Logistic regression
r"""
## <a id="random_forest"></a> Random forest

!!!

### Parameters optimization

!!!

No cos time here because trees capture well non-linear relationships
"""

## column preprocessing
cat_vars = ['month', 'weekday', 'state']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['amt', 'day_time']
tree_col_preproc = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), cat_vars),
     ('id_', FunctionTransformer(None, feature_names_out='one-to-one'),
      bool_vars + quant_vars)])


## Grid search of the regularization parameter with cross validation
scoring = ('precision',  'recall', 'f1')
max_depths = np.arange(2, 17, 2)
rfc = RandomForestClassifier(criterion='gini',
                             max_features=0.5,
                             bootstrap=True,
                             random_state=1234)
clf = Pipeline(
    [('column_preprocessing', tree_col_preproc),
     ('classifier', GridSearchCV(rfc, param_grid={'max_depth': max_depths},
                                 scoring=scoring, refit=False, cv=5))]
)
clf.fit(X_tr, y_tr)


## Extract the relevant metrics
cv_res = clf['classifier'].cv_results_
prec = cv_res['mean_test_precision']
std_prec = cv_res['std_test_precision']
recall = cv_res['mean_test_recall']
std_recall = cv_res['std_test_recall']
f1 = cv_res['mean_test_f1']
std_f1 = cv_res['std_test_f1']


# %%
##
fig6, ax6 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6, 4), dpi=120,
    gridspec_kw={'left': 0.11, 'right': 0.95, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.22})
fig6.suptitle("Figure 6: Scoring metrics vs random forest regularization parameter",
              x=0.02, ha='left')

# RMSE and MAE
ax6.plot(max_depths, prec, color='C0', label='Precision')
ax6.fill_between(max_depths, prec-std_prec, prec+std_prec,
                 color='C0', alpha=0.2)
ax6.plot(max_depths, recall, color='C1', label='Recall')
ax6.fill_between(max_depths, recall-std_recall, recall+std_recall,
                 color='C1', alpha=0.2)
ax6.plot(max_depths, f1, color='C2', label='F1-score')
ax6.fill_between(max_depths, f1-std_f1, f1+std_f1,
                 color='C2', alpha=0.2)
ax6.set_xlim(0, 16)
ax6.set_xticks(np.arange(0, 18, 2))
ax6.set_xlabel('Max depth')
ax6.set_ylim(0, 1)
ax6.set_ylabel('Price error')
ax6.grid(visible=True, linewidth=0.3)
ax6.grid(visible=True, which='minor', linewidth=0.2)
ax6.legend(loc=(0.3, 0.02), ncols=3)


plt.show()

r"""
Figure 6 presents
!!!
"""

# %%

r"""
### Model training and evaluation

According to the previous results, we retain a value $\alpha = 100$ for
the regularization parameter of our ridge model.
"""

## Complete pipeline
randforest_model = Pipeline(
    [('column_preprocessing', col_preproc),
     ('classifier', RandomForestClassifier(criterion='gini',
                                           max_features=0.5,
                                           bootstrap=True,
                                           random_state=1234))]
)

## train
randforest_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = randforest_model.predict(X_tr)
_, prec, recall, f1 = eval_metrics(y_tr, y_pred_tr)
evaluation_df.iloc[2] = prec, recall, f1

## evaluate on test set
y_pred_test = randforest_model.predict(X_test)
_, prec, recall, f1 = eval_metrics(y_test, y_pred_test)
evaluation_df.iloc[3] = prec, recall, f1

print(evaluation_df.loc['RF'])

r"""
!!!
"""



# %% Gradient Boosting model
"""
## <a id="hgbm"></a> Hist Gradient Boosting model

We complete the pricing models catalogue of our application by training
a gradient boosting regressor.


### Parameters optimization

"""

## column preprocessing
cat_vars = ['month', 'weekday', 'state']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['amt', 'day_time']
tree_col_preproc = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), cat_vars),
     ('id_', FunctionTransformer(None, feature_names_out='one-to-one'),
      bool_vars + quant_vars)])


## Grid search of the regularization parameter with cross validation
scoring = ('precision',  'recall', 'f1')
n_est_vals = np.arange(100, 1001, 100)
clf = Pipeline(
    [('column_preprocessing', tree_col_preproc),
     ('classifier', GridSearchCV(HistGradientBoostingClassifier(random_state=1234),
                                 param_grid={'n_estimators': n_est_vals},
                                 scoring=scoring, refit=False, cv=5))]
)
clf.fit(X_tr, y_tr)


## Extract the relevant metrics
cv_res = clf['classifier'].cv_results_
prec = cv_res['mean_test_precision']
std_prec = cv_res['std_test_precision']
recall = cv_res['mean_test_recall']
std_recall = cv_res['std_test_recall']
f1 = cv_res['mean_test_f1']
std_f1 = cv_res['std_test_f1']


#%%

fig7, ax7 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6, 4), dpi=120,
    gridspec_kw={'left': 0.11, 'right': 0.95, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.22})
fig7.suptitle("Figure 7: Scoring metrics vs number of gradient boosting estimators",
              x=0.02, ha='left')


ax7.plot(n_est_vals, prec, color='C0', label='Precision')
ax7.fill_between(n_est_vals, prec-std_prec, prec+std_prec,
                 color='C0', alpha=0.2)
ax7.plot(n_est_vals, recall, color='C1', label='Recall')
ax7.fill_between(n_est_vals, recall-std_recall, recall+std_recall,
                 color='C1', alpha=0.2)
ax7.plot(n_est_vals, f1, color='C2', label='F1-score')
ax7.fill_between(n_est_vals, f1-std_f1, f1+std_f1,
                 color='C2', alpha=0.2)
ax7.set_xlim(0, 1000)
ax7.set_xticks(np.arange(0, 1000, 200))
ax7.set_xticks(np.arange(100, 1000, 200), minor=True)
ax7.set_xlabel('Number of estimators')
ax7.set_ylim(0, 1)
ax7.set_ylabel('Price error')
ax7.grid(visible=True, linewidth=0.3)
ax7.grid(visible=True, which='minor', linewidth=0.2)
ax7.legend(loc=(0.015, 0.9), ncols=3)


plt.show()

r"""
Figure 7 presents plots of our selected metrics as a function of the number of estimators
used with the gradient boosting regressor. The metrics reach an optimum for `n_estimators = 500`,
which we retain for the model.
"""


# %%
"""
### Model training and evaluation
"""

## Complete pipeline
hgb_model = Pipeline(
    [('column_preprocessing', col_preproc),
     ('regressor', HistGradientBoostingClassifier(n_estimators=500,
                                                  random_state=1234))]
)

## train
hgb_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = hgb_model.predict(X_tr)
_, prec, recall, f1 = eval_metrics(y_tr, y_pred_tr)
evaluation_df.iloc[5] = prec, recall, f1

## evaluate on test set
y_pred_test = hgb_model.predict(X_test)
_, prec, recall, f1 = eval_metrics(y_test, y_pred_test)
evaluation_df.iloc[5] = prec, recall, f1

print(evaluation_df['HGB'])

"""
The gradient boosting model is more performant than the simple ridge model.
However, overfitting is still present, and even worse than for ridge regression.
Nevertheless, the test set metrics are quite close
to those obtained by cross-validation (see figure 6).
"""

# %%
"""
## <a id="summary"></a> Summary

"""