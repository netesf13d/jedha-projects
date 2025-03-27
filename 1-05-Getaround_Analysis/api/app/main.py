# -*- coding: utf-8 -*-
"""
FastAPI application.
"""

import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse

from .utils import check_environment_vars, fetch_models, fetch_categories


# =============================================================================
#
# =============================================================================

class PredictionFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


prediction_examples = [
    {
         'model_key': 'Audi',
         'mileage': 132979,
         'engine_power': 112,
         'fuel': 'diesel',
         'paint_color': 'brown',
         'car_type': 'estate',
         'private_parking_available': True,
         'has_gps': True,
         'has_air_conditioning': False,
         'automatic_car': False,
         'has_getaround_connect': True,
         'has_speed_regulator': True,
         'winter_tires': True
    },
]


# =============================================================================
# Setup application
# =============================================================================

models = {}
categories = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialization and shutdown of the application.
    """
    # done at startup
    print('Loading models...', end=' ')
    check_environment_vars()
    models.update(fetch_models(os.environ['MLFLOW_TRACKING_URI']))
    categories.update(fetch_categories(models))
    print('Done.')

    # yield to the app
    yield

    # done at shutdown
    print('wow! Nice application stopping')
    models.clear()


# =============================================================================
#
# =============================================================================

title = 'Getaroun car rental pricing API'

description = (
    'The API interfaces the pricing models registry. It is used by the '
    'dashboard to propose a user-friendly pricing estimation.'
)

tags_metadata = [
    {'name': 'Test',
     'description': 'API test endpoint.'},
    {'name': 'Models info',
     'description': 'Information about available pricing models.'},
    {'name': 'Pricing',
     'description': 'Car rental pricing optimization.'},
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



@app.get('/pricing_models', tags=['Models info'])
async def get_pricing_models()-> list[str]:
    """
    Get the pricing models available from the MLFlow server.
    """
    return list(models)


@app.get('/categories', tags=['Models info'])
async def get_categories()-> dict[str, list[str]]:
    """
    Get the categories associated to each categorical feature.
    """
    return categories


@app.post('/predict/{model_name}', tags=['Pricing'])
async def predict(model_name: str,
                  data: Annotated[PredictionFeatures,
                                  Body(examples=prediction_examples)]
                  )-> dict[str, float]:
    """
    Evaluate a car rental price using the selected model.
    """
    input_features = pd.DataFrame.from_records([dict(data)])

    try:
        model = models[model_name]
    except KeyError:
        detail = {'message': f'pricing model {model_name} not available',
                  'pricing_models': list(models)}
        raise HTTPException(status_code=404, detail=detail)
    else:
        prediction = model.predict(input_features)

    return {'prediction': prediction.tolist()[0]}