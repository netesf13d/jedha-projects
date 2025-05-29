# -*- coding: utf-8 -*-
"""
FastAPI application.
TODO doc
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse

from .utils import check_environment_vars, fetch_models


# =============================================================================
#
# =============================================================================

class PredictionFeatures(BaseModel):
    month: int
    weekday: int
    day_time: int
    amt: float
    category: str
    cust_fraudster: bool
    merch_fraud_victim: bool
    cos_day_time: float
    sin_day_time: float


prediction_examples = [
    {
         'month': 12,
         'weekday': 5,
         'day_time': 85664,
         'amt': 419.52,
         'category': 'entertainment',
         'cust_fraudster': True,
         'merch_fraud_victim': True,
         'cos_day_time': 0.998568,
         'sin_day_time': -0.053498,
    },
]


# =============================================================================
# Setup application
# =============================================================================

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialization and shutdown of the application.
    """
    # done at startup
    print('Loading models...', end=' ')
    check_environment_vars()
    models.update(fetch_models(os.environ['MLFLOW_TRACKING_URI']))
    print('Done.')

    # yield to the app
    yield

    # done at shutdown
    print('wow! Nice application stopping')
    models.clear()


# =============================================================================
#
# =============================================================================

title = 'Automatic fraud detection API'

description = (
    'The API interfaces the fraud detection models registry. It is used by the '
    'fraud detection engine to determine whether a transaction is fraudulent.'
)

tags_metadata = [
    {'name': 'Test',
     'description': 'API test endpoint.'},
    {'name': 'Models info',
     'description': 'Information about available fraud detection models.'},
    {'name': 'Fraud detection',
     'description': 'Fraud detection model.'},
]

contact = {'name': 'Jedha', 'url': 'https://jedha.co'}


app = FastAPI(
    title=title,
    description=description,
    version='0.1',
    contact=contact,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)



@app.get('/', include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.get('/test', tags=['Test'])
async def test()-> int:
    """
    Probe API endpoint.
    """
    return 1 if models else 0



@app.get('/fraud_detection_models', tags=['Models info'])
async def get_pricing_models()-> list[str]:
    """
    Get the pricing models available from the MLflow server.
    """
    return list(models)


@app.post('/predict/{model_name}', tags=['Fraud detection'])
async def predict(model_name: str,
                  data: Annotated[PredictionFeatures,
                                  Body(examples=prediction_examples)]
                  )-> dict[str, int]:
    """
    !!!
    Evaluate a car rental price using the selected model.
    """
    input_features = pd.DataFrame.from_records([dict(data)])

    try:
        model = models[model_name]
    except KeyError:
        detail = {'message': f'fraud detection model {model_name} not available',
                  'models': list(models)}
        raise HTTPException(status_code=404, detail=detail)
    else:
        prediction = model.predict(input_features)

    return {'prediction': prediction.tolist()[0]}