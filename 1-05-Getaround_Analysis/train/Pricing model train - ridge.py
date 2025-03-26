# -*- coding: utf-8 -*-
"""
Script to train a ridge regression model for car rental pricing optimization.
The model is monitored and saved with MLFlow.
"""

import os

import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             root_mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import Ridge


## Setup environment variables
os.environ['MLFLOW_TRACKING_URI'] = 'https://netesf13d-mlflow-server-1-05-getaround.hf.space/'
S3_WRITER_ACCESS_KEYS = './certification-project-s3-writer_accessKeys.key'
with open(S3_WRITER_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    id_, key_ = f.readlines()[-1].strip().split(',')
os.environ["AWS_ACCESS_KEY_ID"] = id_
os.environ["AWS_SECRET_ACCESS_KEY"] = key_


## Check that the required env variables are set
if 'MLFLOW_TRACKING_URI' not in os.environ:
    raise KeyError('environment variable `MLFLOW_TRACKING_URI` is not set')
if 'AWS_ACCESS_KEY_ID' not in os.environ:
    raise KeyError('artifact store environment variable '
                   '`AWS_ACCESS_KEY_ID` is not set')
if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
    raise KeyError('artifact store environment variable '
                   ' `AWS_SECRET_ACCESS_KEY` is not set')


## Configure experiment
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
experiment = mlflow.set_experiment('car-pricing-ridge-model')
mlflow.set_experiment_tag('model_category', 'linear')


## Configure logging
mlflow.autolog(disable=True)
# mlflow.sklearn.autolog() # Autologging functionality


random_state = 1234

# =============================================================================
# Run experiment
# =============================================================================

## Load and prepare data
df = pd.read_csv('./get_around_pricing_project.csv')
df = df.drop('Unnamed: 0', axis=1)
df = df.astype({'mileage': float,
                'engine_power': float,
                'rental_price_per_day': float})


## Remove outliers
q1, q2 = 0.25, 0.75
k = 6 # exclusion factor
quantiles = df.loc[:, ['mileage', 'rental_price_per_day']].quantile([q1, q2])
iqr_mil, iqr_pr = quantiles.iloc[1] - quantiles.iloc[0]
med_mil, med_pr = df.loc[:, ['mileage', 'rental_price_per_day']].median()
loc = ((df['mileage'] < med_mil + k*iqr_mil)
       * (df['rental_price_per_day'] < med_pr + k*iqr_pr))
processed_df = df.loc[loc]


## Create instance of PandasDataset for logging
# dataset = mlflow.data.from_pandas(processed_df,
#                                   name='getaround_delay_analysis',
#                                   targets='rental_price_per_day')


## data preparation
y = processed_df['rental_price_per_day']
X = processed_df.drop('rental_price_per_day', axis=1)

## train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state)

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


with mlflow.start_run(experiment_id=experiment.experiment_id):
    ## full pipeline
    alpha = 1e2
    model = Pipeline([('column_preprocessing', col_preproc),
                      ('regressor', Ridge(alpha=alpha))])

    ## Fit
    model.fit(X_tr, y_tr)

    ## Log dataset
    # mlflow.log_input(dataset, context='training')

    ## Log parameters manually
    mlflow.log_param('random_state', random_state)
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('steps', model.steps)
    ## Log model
    mlflow.sklearn.log_model(model, 'ridge',
                             registered_model_name='ridge-regression',
                             input_example=df.iloc[[0], :-1],
                             metadata={'description': 'Ridge regression'})
    ## Log categories
    ohe = model['column_preprocessing']['cat_ohe']
    categories = {k: v.tolist() for k, v in zip(ohe.feature_names_in_, ohe.categories_)}
    mlflow.log_param('categories', categories)

    ## Evaluate of train set
    y_pred_tr = model.predict(X_tr)
    mlflow.log_metric('train-mse', mean_squared_error(y_tr, y_pred_tr))
    mlflow.log_metric('train-rmse', root_mean_squared_error(y_tr, y_pred_tr))
    mlflow.log_metric('train-r-squared', r2_score(y_tr, y_pred_tr))
    mlflow.log_metric('train-mae', mean_absolute_error(y_tr, y_pred_tr))
    mlflow.log_metric('train-mape',
                      100*mean_absolute_percentage_error(y_tr, y_pred_tr))

    ## Evaluate on test set
    y_pred_test = model.predict(X_test)
    mlflow.log_metric('test-mse', mean_squared_error(y_test, y_pred_test))
    mlflow.log_metric('test-rmse', root_mean_squared_error(y_test, y_pred_test))
    mlflow.log_metric('test-r-squared', r2_score(y_test, y_pred_test))
    mlflow.log_metric('test-mae', mean_absolute_error(y_test, y_pred_test))
    mlflow.log_metric('test-mape',
                      100*mean_absolute_percentage_error(y_test, y_pred_test))
