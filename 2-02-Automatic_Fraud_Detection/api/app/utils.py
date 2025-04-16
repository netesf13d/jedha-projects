# -*- coding: utf-8 -*-
"""
API utilities
"""

import ast
import os

import mlflow



def check_environment_vars()-> None:
    """
    Check that the following environment variables are set in order to
    interact with the MLflow server and fetch the models:
    - `MLFLOW_TRACKING_URI`: the URI of MLflow tracking server
    - `AWS_ACCESS_KEY_ID`: access id to the artifact store
    - `AWS_SECRET_ACCESS_KEY`: access key to the artifact store
    """
    if 'MLFLOW_TRACKING_URI' not in os.environ:
        raise KeyError('environment variable `MLFLOW_TRACKING_URI` is not set')
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        raise KeyError('artifact store environment variable '
                       '`AWS_ACCESS_KEY_ID` is not set')
    if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        raise KeyError('artifact store environment variable '
                       ' `AWS_SECRET_ACCESS_KEY` is not set')


def fetch_models(mlflow_tracking_uri: str)-> dict:
    """
    Fetch available pricing optimization models from the MLflow server.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_infos = mlflow.search_registered_models()
    model_uris = {m.name: f'models:/{m.name}/{m.latest_versions[0].version}'
                  for m in model_infos}
    return {name: mlflow.pyfunc.load_model(uri)
              for name, uri in model_uris.items()}


def fetch_categories(models: dict)-> dict[str, list[str]]:
    """

    """
    model = next(iter(models.values()))
    run = mlflow.get_run(model._model_meta.run_id)
    return ast.literal_eval(run.data.params['categories'])