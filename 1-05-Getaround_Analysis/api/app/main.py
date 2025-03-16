# -*- coding: utf-8 -*-
"""

"""

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from pydantic import BaseModel

# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import (mean_squared_error,
#                              r2_score,
#                              mean_absolute_error,
#                              mean_absolute_percentage_error)
# from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import (OneHotEncoder,
#                                    StandardScaler,
#                                    FunctionTransformer)
# from sklearn.linear_model import Ridge
# from sklearn.svm import SVR

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.openapi.docs import get_swagger_ui_html

# from .utils import iter_npz
# from .routers import fetching, processing


# CURR_MODEL = 'runs:/c09d09ef14e546b08f2f339d2c966da6/salary_estimator'

# =============================================================================
# setup model
# =============================================================================

# ## column preprocessing
# cat_vars = ['model_key', 'fuel', 'paint_color', 'car_type']
# bool_vars = ['private_parking_available', 'has_gps', 'has_air_conditioning',
#              'automatic_car', 'has_getaround_connect', 'has_speed_regulator',
#              'winter_tires']
# quant_vars = ['mileage', 'engine_power']
# col_preproc = ColumnTransformer(
#     [('cat_ohe',
#       OneHotEncoder(drop=None, handle_unknown='infrequent_if_exist', min_frequency=0.01),
#       cat_vars),
#      ('bool_id', FunctionTransformer(feature_names_out='one-to-one'), bool_vars),
#      ('quant_scaler', StandardScaler(), quant_vars)])

# col_preproc

# ##
# ridge_model = Pipeline([('column_preprocessing', col_preproc),
#                         ('regressor', Ridge())])


models = {'ridge': lambda x: 100., 'SVM': lambda x: 100.}


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


# =============================================================================
# 
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    TODO
    - load model from config file
    - put model in app namespace
    """
    # done at startup
    print('wow! Nice application launching')
    # model = mlflow.pyfunc.load_model(CURR_MODEL)
    # yield to the app
    yield
    # done at shutdown
    print('wow! Nice application stopping')
    

# =============================================================================
# 
# =============================================================================

title = 'Getaround rental pricing API'

description = "Nice place to be"

tags_metadata = [
    {'name': 'Test',
     'description': 'API test endpoint.'},
    {'name': 'Prediction',
     'description': 'Prediction endpoint.'}
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

# app.mount('/', StaticFiles(directory='./static', html=True), name='static')
# app.include_router(fetching.router)
# app.include_router(processing.router)


@app.get('/test', tags=['Test'])
async def test()-> int:
    """
    Probe API endpoint.
    """
    return 1


@app.get('/features', tags=['Test'])
async def get_car_models()-> dict[str, str]:
    """
    
    """
    return 


@app.get('/categories', tags=['Test'])
async def get_categories()-> dict[str, list[str]]:
    """
    
    """
    return ['a']


@app.get('/car_models', tags=['Test'])
async def get_car_models()-> list[str]:
    """
    
    """
    return ['a']


# @app.post('/predict', tags=['Machine Learning'])
# async def dummy_predict(model: str):
#     """
#     Return optimal rental price.
#     """
#     return model


@app.post('/predict', tags=['Machine Learning'])
async def predict(data: PredictionFeatures): #, model: str):
    """
    Return optimal rental price.
    """
    model = 'ridge'
    input_features = pd.DataFrame(dict(data))

    # model = mlflow.pyfunc.load_model(CURR_MODEL)
    try:
        model = models[model]
    except KeyError:
        raise KeyError(f'`model` must be in {set(models.keys())}')
    else:
        prediction = model.predict(input_features)
    
    # format response
    response = {'prediction': prediction.tolist()[0]}
    
    return response
















