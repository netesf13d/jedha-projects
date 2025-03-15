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

print(df.describe(include='all'))

"""
The dataset has 4843 rows, which correspond to distinct cars available on the platform.

weird value for mileage

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
print(df['car_type'].value_counts())


print(df['fuel'].value_counts())


print(df['paint_color'].value_counts())


print(df['model_key'].value_counts())

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

# %%

# cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']
bool_vars = ['private_parking_available', 'has_gps', 'has_air_conditioning',
             'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
             'winter_tires']
quant_vars = ['mileage', 'engine_power']

df_ = df.loc[:, ['rental_price_per_day'] + quant_vars + bool_vars]
corr_df = df_.corr().to_numpy()


fig2, ax2 = plt.subplots(
    nrows=1, ncols=1, figsize=(6.2, 5.6), dpi=100,
    gridspec_kw={'left': 0.14, 'right': 0.9, 'top': 0.76, 'bottom': 0.04, 'wspace': 0.24})
cax2 = fig2.add_axes((0.88, 0.04, 0.03, 0.72))
fig2.suptitle("Figure 2: Correlation matrix",
              x=0.02, ha='left')


ax2.set_aspect('equal')
cmap2 = plt.get_cmap('coolwarm').copy()
cmap2.set_extremes(under='0.9', over='0.5')
heatmap = ax2.pcolormesh(corr_df[::-1]*100, cmap=cmap2, vmin=-70, vmax=70,
                             edgecolors='0.2', linewidth=0.5)

ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# ax2.tick_params(axis='y', pad=15)
ticklabels = ['Rental price', 'Mileage', 'Engine power',
              'Priv. parking', 'GPS', 'Air cond.', 'Automatic car',
              'Getaround conn.', 'Speed reg.', 'Winter tires']
ax2.set_xticks(np.linspace(0.5, 9.5, 10), ticklabels, rotation=50, ha='left')
ax2.set_yticks(np.linspace(9.5, 0.5, 10), ticklabels, rotation=0)
# ax2.set_yticks(np.linspace(0.5, 5.5, 6), attributes[::-1])

for (i, j), c in np.ndenumerate(corr_df[::-1]*100):
    ax2.text(j+0.52, i+0.45, f'{c:.1f}', ha='center', va='center', fontsize=9)

# ax2.set_title('Females', fontweight='bold')
pos = ax2.get_position().bounds
x, y = pos[0], pos[1] + pos[3]

fig2.colorbar(heatmap, cax=cax2, orientation="vertical", ticklocation="right")
cax2.set_title('Correlation coef. (%)', y=-3.4)


plt.show()

"""
Figure 2
The rental price is strongly correlated to the engine power, and in general with
all additional car options, except the prsence of winter tires. On the other hand,
a higher mileage reflects into a lower rental price.
"""

# %%

cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']

df_ = df.loc[:, cat_vars]
value_counts = df_.apply(lambda x: x.map(x.value_counts()))
df_ = pd.concat([df_.mask(value_counts < 30, 'other'),
                 df['rental_price_per_day']], axis=1)


fig3, axs3 = plt.subplots(
    nrows=4, ncols=1, figsize=(6, 8.8), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.975, 'top': 0.95, 'bottom': 0.06,
                 'hspace': 0.34})
fig3.suptitle("Figure 3: ",
              x=0.02, ha='left')


for item, ax in zip(cat_vars, axs3):
    vdata, categories = [], []
    for category, gdf in df_.loc[:, ['rental_price_per_day', item]].groupby(item):
        vdata.append(gdf['rental_price_per_day'].to_numpy())
        categories.append(category)
    
    n = len(categories)
    vplot = ax.violinplot(vdata, positions=np.linspace(0, n-1, n),
                          widths=0.8, showmeans=True)
    for elt in vplot['bodies']:
        elt.set_facecolor('plum')
    for elt in ['cbars', 'cmaxes', 'cmins', 'cmeans']:
        vplot[elt].set_edgecolor('plum')
    ax.set_xticks(np.arange(0, len(categories)), categories, fontsize=10,
                  rotation=30, ha='right')
    ax.set_ylim(0, 450)
    ax.grid(visible=True, axis='y', linewidth=0.3)


fig3.text(0.02, 0.51, 'Rental price ($/day)', rotation=90, fontsize=11,
          ha='center', va='center')
    

plt.show()

"""
Figure 3
"""


# %%


rental_price = df['rental_price_per_day'].to_numpy()
rp_bins = np.linspace(0, 450, 91)

mileage = df['mileage'].to_numpy()
m_bins = np.linspace(0, 1_050_000, 106)

engine_power = df['engine_power'].to_numpy()
ep_bins = np.linspace(0, 450, 46)


##
fig4, axs4 = plt.subplots(
    nrows=1, ncols=2, figsize=(8.8, 4.2), dpi=100,
    gridspec_kw={'left': 0.07, 'right': 0.89, 'top': 0.88, 'bottom': 0.12,
                 'wspace': 0.18})
cax4 = fig4.add_axes((0.915, 0.12, 0.025, 0.76))
fig4.suptitle("Figure 4: ",
              x=0.02, ha='left')



cmap4 = plt.get_cmap('hot').copy()
cmap4.set_extremes(under='0.7', over='0.5')
# axs4[0].set_aspect('equal')

heatmap = axs4[0].pcolormesh(
    *np.meshgrid(m_bins, rp_bins, indexing='ij'),
    np.histogram2d(mileage, rental_price, bins=[m_bins, rp_bins])[0],
    cmap=cmap4, vmin=0.5, vmax=50)
# axs4[0].set_ylim(0, 450)

axs4[0].set_xticks(np.linspace(1e5, 1e6, 10), minor=True)
axs4[0].set_xlabel('Mileage', fontsize=11)
axs4[0].set_yticks(np.arange(50, 500, 100), minor=True)
axs4[0].set_ylabel('Rental price', fontsize=11)
axs4[0].set_title('Mileage x Rental price')

heatmap = axs4[1].pcolormesh(
    *np.meshgrid(ep_bins, rp_bins, indexing='ij'),
    np.histogram2d(engine_power, rental_price, bins=[ep_bins, rp_bins])[0],
    cmap=cmap4, vmin=0.5, vmax=200)
axs4[1].set_xlim(0, 450)
axs4[1].set_xticks(np.arange(50, 500, 100), minor=True)
axs4[1].set_xlabel('Engine power', fontsize=11)
axs4[1].set_title('Engine power x Rental price')

fig4.colorbar(heatmap, cax=cax4, orientation="vertical", ticklocation="right")
cax4.set_title('Correlation coef. (%)', y=-3.4)
cax4.set_yticks(np.linspace(0, 200, 9))

plt.show()

"""
Figure 4
"""


# %% Ridge

"""
## <a id="utils"></a> Utilities

###
"""

# TODO remove outliers


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



# %%
"""
## <a id="conclusion"></a> Conclusion

"""

evaluation_df

"""

"""

