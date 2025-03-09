# -*- coding: utf-8 -*-
"""
Script for Getaround project.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.base import clone
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
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# %% Loading
"""
## <a name="loading"></a> Data loading
"""

##
df = pd.read_csv('../data/get_around_pricing_project.csv')
df = df.drop('Unnamed: 0', axis=1)

print(df.describe(include='all'))

"""
The dataset has 4843 rows, which correspond to distinct cars available on the platform.

The dataset is split in two parts, with 21310 observations each.
- The first dataframe, `pp_df`, contains the information that we have *before* a car is rent.
We will use these features as predictors for our machine learning model
- The second dataframe contains information relative to the *outcome* of a rental. This data cannot be used to make predictions.
However, we will use it to build target feature to train our ML model.

We should include all the data available in our exploratory data analysis. we build a combined dataframe `eda_df` for this purpose.
"""

# eda_df = pd.concat([da_df, pp_df], axis=1)
# eda_df


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
df['car_type'].value_counts()


"""
### some graphs


- follow the trajectory of a car
"""


#%%

cols = ['private_parking_available', 'has_gps', 'has_air_conditioning',
        'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
        'winter_tires']

fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(7, 4.2), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.975, 'top': 0.91, 'bottom': 0.11, 'hspace': 0.08})
fig1.suptitle("Figure 1: Comparison of rental prices vs various car attributes",
              x=0.02, ha='left')


vdata, vpos = [], []
vdata_true, vdata_false = [], []
for i, item in enumerate(cols):
    for is_, gdf in df.groupby(item):
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

# %% Ridge

"""
## <a id="utils"></a> Utilities

###
"""

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
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
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

# 'mean_test_neg_mean_absolute_error',
# 'mean_test_neg_mean_absolute_percentage_error',
# 'mean_test_neg_mean_squared_error',
# 'mean_test_neg_root_mean_squared_error',
# 'mean_test_r2',



# ## setup cross-validated grid search
# reg = clone(ridge_model)
# alphas = np.logspace(-1, 5, 61)
# cv = KFold(n_splits=10, shuffle=True, random_state=1234)

# ## Run the search
# metrics = np.zeros((5, len(alphas)))
# for i, alpha in enumerate(alphas):
#     reg['regressor'].alpha = alpha
    
#     # construct validation predictions vector to compute validation metrics
#     y_pred_v = np.zeros(len(y_tr), dtype=float)
#     for itr, ival in cv.split(X_tr, y_tr):
#         reg.fit(X_tr.iloc[itr], y_tr.iloc[itr])
#         y_pred_v[ival] = reg.predict(X_tr.iloc[ival])
#     metrics[:, i] = eval_metrics(y_tr, y_pred_v)


"""
### Model evaluation

"""
















