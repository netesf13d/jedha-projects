# -*- coding: utf-8 -*-
"""

"""

import io
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.openapi.docs import get_swagger_ui_html

# from .utils import iter_npz
# from .routers import fetching, processing


CURR_MODEL = 'runs:/c09d09ef14e546b08f2f339d2c966da6/salary_estimator'

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
    model = mlflow.pyfunc.load_model(CURR_MODEL)
    # yield to the app
    yield
    # done at shutdown
    print('wow! Nice application stopping')
    

# =============================================================================
# 
# =============================================================================

title = "Getaround rental pricing API"

description = "Nice place to be"

tags_metadata = [
    {"name": "Prediction",
     "description": "Prediction Endpoint."}
]

contact = {"name": "Jedha", "url": "https://jedha.co"}


app = FastAPI(
    title=title,
    description=description,
    version="0.1",
    contact=contact,
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# app.mount('/', StaticFiles(directory='./static', html=True), name='static')
# app.include_router(fetching.router)
# app.include_router(processing.router)


@app.get("/hello", tags=["test"])
async def probe()-> int:
    """
    Probe API endpoint.
    """
    return 1




class PredictionFeatures(BaseModel):
   test: float




@app.post("/predict", tags=["Machine Learning"])
async def predict(data: PredictionFeatures):
    """
    Return optimal rental price.
    """
    input_features = pd.DataFrame(dict(data))

    model = mlflow.pyfunc.load_model(CURR_MODEL)

    prediction = model.predict(input_features)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response


