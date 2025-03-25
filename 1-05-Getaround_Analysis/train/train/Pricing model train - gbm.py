# -*- coding: utf-8 -*-
"""
Script to train a gradient boosting model for car rental pricing optimization.
The model is monitored and saved with MLFlow.
"""

import os

import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.ensemble import GradientBoostingRegressor


# Set your variables for your environment
EXPERIMENT_NAME = 'car-pricing-gradient-boosting-model'

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


# =============================================================================
# 
# =============================================================================

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


# =============================================================================
# 
# =============================================================================

## Load and prepare data
df = pd.read_csv('./data/get_around_pricing_project.csv')
df = df.drop('Unnamed: 0', axis=1)

## Remove outliers
q1, q2 = 0.25, 0.75
k = 6 # exclusion factor
quantiles = df.loc[:, ['mileage', 'rental_price_per_day']].quantile([q1, q2])
iqr_mil, iqr_pr = quantiles.iloc[1] - quantiles.iloc[0]
med_mil, med_pr = df.loc[:, ['mileage', 'rental_price_per_day']].median()
loc = ((df['mileage'] < med_mil + k*iqr_mil)
       * (df['rental_price_per_day'] < med_pr + k*iqr_pr))
processed_df = df.loc[loc]

## data preparation
y = processed_df['rental_price_per_day']
X = processed_df.drop('rental_price_per_day', axis=1)

## train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

## column preprocessing
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


with mlflow.start_run(experiment_id = experiment.experiment_id):
    ## full pipeline
    nest = 500
    gb_model = Pipeline(
        [('column_preprocessing', col_preproc),
         ('regressor', GradientBoostingRegressor(n_estimators=nest, random_state=1234))]
    )
    
    ## Fit
    gb_model.fit(X_tr, y_tr)
    
    
    ## evaluate of train set
    y_pred_tr = gb_model.predict(X_tr)
    tr_mse = mean_squared_error(y_tr, y_pred_tr)
    tr_rmse = mean_squared_error(y_tr, y_pred_tr)
    tr_r2 = r2_score(y_tr, y_pred_tr)
    tr_mae = mean_absolute_error(y_tr, y_pred_tr)
    tr_mape = 100 * mean_absolute_percentage_error(y_tr, y_pred_tr)
    # tr_metrics = eval_metrics(y_tr, y_pred_tr)
    
    ## evaluate on test set
    y_pred_test = gb_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mape = 100 * mean_absolute_percentage_error(y_test, y_pred_test)
    # test_metrics = eval_metrics(y_test, y_pred_test)
