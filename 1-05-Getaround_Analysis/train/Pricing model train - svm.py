# -*- coding: utf-8 -*-
"""
Script for Getaround project.
"""

# from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


# %% Loading
"""
## <a name="loading"></a> Data loading
"""

##
df = pd.read_csv('../data/get_around_pricing_project.csv')
df = df.drop('Unnamed: 0', axis=1)




## data preparation
y = df['rental_price_per_day']
X = df.drop('rental_price_per_day', axis=1) # Date is redundant with (Year, Month, Week)

## train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

## setup pipeline
# column preprocessing
cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']
bool_vars = ['private_parking_available', 'has_gps', 'has_air_conditioning',
             'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
             'winter_tires']
quant_vars = ['mileage', 'engine_power']
col_preproc = ColumnTransformer(
    [('cat_ohe',
      OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.01),
      cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])


"""
### Model evaluation utilities

"""

##
# def scorer(estimator, X, y)

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
    [('Ridge', 'SVM'), ('train', 'test')],
    names=['model', 'eval. set'])
evaluation_df = pd.DataFrame(
    np.full((4, 5), np.nan), index=index, columns=metric_names)


# %% Ridge

"""
## <a id="linreg"></a> Ridge regression


### Model construction and training
"""


## full pipeline
ridge_model = Pipeline([('column_preprocessing', col_preproc),
                        ('regressor', Ridge())])

scoring = ('neg_mean_squared_error',  'neg_root_mean_squared_error', 'r2',
           'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error')
alphas = np.logspace(-1, 5, 61)
reg = Pipeline(
    [('column_preprocessing', col_preproc),
     ('regressor', GridSearchCV(Ridge(), param_grid={'alpha': alphas},
                                scoring=scoring, refit=False, cv=10))]
)


reg.fit(X_tr, y_tr)

reg['regressor'].cv_results_['param_alpha']

# %%

cv_res = reg['regressor'].cv_results_

mse = -cv_res['mean_test_neg_mean_squared_error']
rmse = -cv_res['mean_test_neg_root_mean_squared_error']
r2 = cv_res['mean_test_r2']
mae = -cv_res['mean_test_neg_mean_absolute_error']
mape = -cv_res['mean_test_neg_mean_absolute_percentage_error']


fig5, axs5 = plt.subplots(
    nrows=1, ncols=3, sharex=True, sharey=False, figsize=(9, 3.6), dpi=120,
    gridspec_kw={'left': 0.07, 'right': 0.98, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.22})
fig5.suptitle("Figure 5: Metrics vs the regularization parameter",
              x=0.02, ha='left')

# RMSE and MAE
axs5[0].plot(alphas, rmse, color='C0',
             label='Root mean sq. err.')
axs5[0].plot(alphas, mae, color='C1',
             label='Mean abs. err')
# axs5[0].axvline(best_alpha, color='k', linewidth=0.6)

axs5[0].set_xscale('log')
axs5[0].set_xlim(1e-1, 1e5)
axs5[0].set_xticks(np.logspace(-1, 5, 7))
axs5[0].set_xticks(
    np.concatenate([np.linspace(10**i, 10**(i+1), 10) for i in range(-1, 5)]),
    minor=True)
axs5[0].set_xlabel('regularization parameter')
# axs5[0].set_ylim(0, 8e5)
# axs5[0].set_yticks(np.linspace(0, 8e5, 9),
#                    [f'{i:.1f}' for i in np.linspace(0, 0.8, 9)])
axs5[0].set_ylabel('Error (M$)')
axs5[0].grid(visible=True, linewidth=0.3)
axs5[0].grid(visible=True, which='minor', linewidth=0.2)
axs5[0].legend(loc=(0.015, 0.81))

# R-squared
axs5[1].plot(alphas, r2, color='C2', label='R-squared')
# axs5[1].axvline(best_alpha, color='k', linewidth=0.6)

axs5[1].set_xlabel('regularization parameter')
axs5[1].set_ylim(0, 1)
axs5[1].grid(visible=True, linewidth=0.3)
axs5[1].grid(visible=True, which='minor', linewidth=0.2)
axs5[1].legend(loc=(0.015, 0.81))

# MAPE
axs5[2].plot(alphas, mape, color='C3', label='MAPE')
# axs5[2].axvline(best_alpha, color='k', linewidth=0.6)

axs5[2].set_xlabel('regularization parameter')
# axs5[2].set_ylim(0.7e0, 2e1)
axs5[2].grid(visible=True, linewidth=0.3)
axs5[2].grid(visible=True, which='minor', linewidth=0.2)
axs5[2].legend(loc=(0.015, 0.89))


plt.show()


# %%
"""
### Model evaluation

"""

## train
ridge_model['regressor'].alpha = 1e2
ridge_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = ridge_model.predict(X_tr)
evaluation_df.iloc[0] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = ridge_model.predict(X_test)
evaluation_df.iloc[1] = eval_metrics(y_test, y_pred_test)


evaluation_df.loc['Ridge']

"""
Niceeee
"""
#%%
"""
### Model interpretation
"""

# recover feature names
col_preproc_ = ridge_model['column_preprocessing']
features = ['intercept']
features += [feature_name.split('__')[1].strip('_sklearn')
             for feature_name in col_preproc_.get_feature_names_out()]

# get coefficients
intercept = ridge_model['regressor'].intercept_
coef_vals = np.concatenate([[intercept], ridge_model['regressor'].coef_])


for feature, coef in zip(features, coef_vals):
    print(f'{feature:<24} : {coef:>6.2f}')


# %%
"""
## <a id="svm"></a> Support vector machine


### Model construction and training
"""

## full pipeline
svm_model = Pipeline([('column_preprocessing', col_preproc),
                        ('regressor', SVR())])

scoring = ('neg_mean_squared_error',  'neg_root_mean_squared_error', 'r2',
           'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error')
# alphas = np.logspace(-1, 5, 61)
# reg = Pipeline(
#     [('column_preprocessing', col_preproc),
#      ('regressor', GridSearchCV(Ridge(), param_grid={'alpha': alphas},
#                                 scoring=scoring, refit=False, cv=10))]
# )


# reg.fit(X_tr, y_tr)

# reg['regressor'].cv_results_['param_alpha']


"""
### Model evaluation

"""

## train
svm_model['regressor'].alpha = 1e2
svm_model.fit(X_tr, y_tr)

## evaluate of train set
y_pred_tr = svm_model.predict(X_tr)
evaluation_df.iloc[2] = eval_metrics(y_tr, y_pred_tr)

## evaluate on test set
y_pred_test = svm_model.predict(X_test)
evaluation_df.iloc[3] = eval_metrics(y_test, y_pred_test)


evaluation_df.loc['SVM']


