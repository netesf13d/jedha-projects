# -*- coding: utf-8 -*-
"""
Script to run the API locally.
"""

import os
import uvicorn
from app import app


## MLFLOW tracking server URI
MLFLOW_TRACKING_URI = './mlflow_tracking_uri.key'
with open(MLFLOW_TRACKING_URI, 'rt', encoding='utf-8') as f:
    os.environ['MLFLOW_TRACKING_URI'] = f.read().strip()

## MLFlow artifact store access credentials
S3_WRITER_ACCESS_KEYS = './s3-writer_accessKeys.key'
with open(S3_WRITER_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    id_, key_ = f.readlines()[-1].strip().split(',')
os.environ["AWS_ACCESS_KEY_ID"] = id_
os.environ["AWS_SECRET_ACCESS_KEY"] = key_


## Run app
uvicorn.run(app, host="0.0.0.0", port=8000, log_level='info')