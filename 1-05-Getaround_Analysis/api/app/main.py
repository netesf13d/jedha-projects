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

from .utils import iter_npz
from .routers import fetching, processing

# =============================================================================
# 
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # done at startup
    print('wow! Nice application launching')
    # yield to the app
    yield
    # done at shutdown
    print('wow! Nice application stopping')
    

# =============================================================================
# 
# =============================================================================

title = "Crypto crash prediction API"

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
app.include_router(fetching.router)
app.include_router(processing.router)


@app.get("/hello", tags=["test"])
async def probe()-> int:
    """
    Probe API endpoint.
    """
    return 1


import numpy as np
rng = np.random.default_rng(1234)



@app.get("/test", tags=["test"])
async def test(n: int):
    """
    Nice test
    """
    crash_data = {
        'n': n,
        'duration': rng.poisson(lam=60, size=n),
    }
    
    gen_ = iter_npz(crash_data)
    
    return StreamingResponse(gen_, media_type="zip-archive/npz")



# class PredictionFeatures(BaseModel):
#     YearsExperience: float




# @app.post("/predict", tags=["Machine Learning"])
# async def predict(predictionFeatures: PredictionFeatures, i: int):
#     """
#     Prediction of probabilities of a crash for a given cryptocurrency! 
#     """
#     # Read data 
#     years_experience = pd.DataFrame({"YearsExperience": [predictionFeatures.YearsExperience]})

#     # Log model from mlflow 
#     logged_model = 'runs:/c09d09ef14e546b08f2f339d2c966da6/salary_estimator'

#     # Load model as a PyFuncModel.
#     loaded_model = mlflow.pyfunc.load_model(logged_model)

#     # If you want to load model persisted locally
#     #loaded_model = joblib.load('salary_predictor/model.joblib')

#     prediction = loaded_model.predict(years_experience)

#     # Format response
#     response = {"prediction": prediction.tolist()[0]}
#     return response


