# -*- coding: utf-8 -*-
"""
Script to train a ridge regression model for car rental pricing optimization.
The model is monitored and saved with MLFlow.
"""

import os

import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import LogisticRegression



## Setup environment variables
os.environ['MLFLOW_TRACKING_URI'] = 'https://netesf13d-mlflow-server-2-03-fraud-detection.hf.space/'
# os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:7860'
S3_WRITER_ACCESS_KEYS = './s3-writer_accessKeys.key'
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
experiment = mlflow.set_experiment('fraud-detection')
mlflow.set_experiment_tag('model_category', 'linear')

## Configure logging
mlflow.autolog(disable=True)
# mlflow.sklearn.autolog() # Autologging functionality


# =============================================================================
# Run experiment
# =============================================================================

## Load data
df = pd.read_csv('./fraud_train_data.csv')


## Create instance of PandasDataset for logging
# dataset = mlflow.data.from_pandas(df, name='automatic_fraud_detection',
#                                   targets='is_fraud')

## data preparation
y = df['is_fraud']
X = df.drop(['is_fraud', 'merchant_id', 'customer_id'], axis=1)

## train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1234)


## Column preprocessing
cat_vars = ['weekday', 'category']
bool_vars = ['cust_fraudster', 'merch_fraud_victim']
quant_vars = ['amt', 'cos_day_time', 'sin_day_time']

col_preproc = ColumnTransformer(
    [('cat_ohe',
      OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.005),
      cat_vars),
     ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)]
)


with mlflow.start_run(experiment_id=experiment.experiment_id):
    ## Log dataset
    # mlflow.log_input(dataset, context='training')
    
    ## Full pipeline, fit on train data
    C = 1e3
    model = Pipeline([('column_preprocessing', col_preproc),
                      ('classifier', LogisticRegression(C=C))])
    model.fit(X_tr, y_tr)

    ## Log parameters manually
    mlflow.log_param('C', C)
    mlflow.log_param('steps', model.steps)
    
    ## Evaluate of train set
    y_pred_tr = model.predict(X_tr)
    mlflow.log_metric('train-precision', precision_score(y_tr, y_pred_tr))
    mlflow.log_metric('train-recall', recall_score(y_tr, y_pred_tr))
    mlflow.log_metric('train-f1', f1_score(y_tr, y_pred_tr))

    ## Evaluate on test set
    y_pred_test = model.predict(X_test)
    mlflow.log_metric('test-precision', precision_score(y_test, y_pred_test))
    mlflow.log_metric('test-recall', recall_score(y_test, y_pred_test))
    mlflow.log_metric('test-f1', f1_score(y_test, y_pred_test))
    
    ## Re-train with full dataset and log model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, 'log-reg',
                             registered_model_name='logistic-regression',
                             input_example=X.iloc[[0]],
                             metadata={'description': 'Logistic regression'})