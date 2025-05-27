---
title: Automatic Fraud Detection API
emoji: 💸
sdk: docker
app_port: 7860
---
# Automatic Fraud Detection API

In this project we deploy a small application.


## Contents

The API provides two endpoints:
- `GET test`: Probe the API for a response
- `POST predict`: Get a car rental price from car attributes
- The `requirements.txt` files applies to the docker image


## Deployment

### Local deployment



```bash
python run_api.py
```


##### Windows

`gunicorn` is not available on Windows, you must use `uvicorn` to run the app:
```bash
uvicorn app:app --reload --host localhost --port 8000
```

##### Linux/Unix

Both `gunicorn` and `uvicorn` are both available. It is [recommended](https://www.uvicorn.org/deployment/) to use `uvicorn`.
```bash
uvicorn app:app --reload --host localhost --port 8000
```

Using `gunicorn` is also possible, but the dynamic reload of the API will not be enabled.
```bash
gunicorn app:app --host localhost --port 8000 --worker-class uvicorn_worker.UvicornWorker
```


### Deployment on HuggingFace

It is assumed that the deployment OS is Linux.
Port 8501 is required by HuggingFace
```bash
gunicorn --bind 0.0.0.0:8501 --worker-class uvicorn_worker.UvicornWorker app:app
```

### Cloud deployment

The tracking server is deployed in an Huggingface space.

1. Create a huggingface [space](https://huggingface.co/new-space). Choose `docker` as the software development kit.
2. Transfer the contents of this directory: `Dockerfile`, `README.md`, `requirements.txt` and `app/`.
3. Setup the variables and secrets in the space's settings
  - Secret `AWS_ACCESS_KEY_ID`,
  - Secret `AWS_SECRET_ACCESS_KEY`,
  - Variable `MLFLOW_TRACKING_URI`,
  - Variable `PORT`. Here the value is set to 7860, the default value (in any case, it must match the `app_port` config variable of the space).
4. Run the space and set it public. It will be accessible at `https://{owner}-{your-space-name}.hf.space/`.


## Usage


```bash
curl -X 'POST' \
  'http://localhost:8000/predict/logistic-regression' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "month": 6,
    "weekday": 6,
    "day_time": 45948,
    "amt": 242.35,
    "category": "health_fitness",
    "cust_fraudster": false,
    "merch_fraud_victim": false,
    "cos_day_time": -0.980098,
    "sin_day_time": -0.198513,
  }'
```


```python
>>> import requests
>>> model_name = 'ridge_regression'
>>> data = {'month': 6,
            'weekday': 6,
            'day_time': 45948,
            'amt': 242.35,
            'category': 'health_fitness',
            'cust_fraudster': False,
            'merch_fraud_victim': False,
            'cos_day_time': -0.980098,
            'sin_day_time': -0.198513}
>>> r = requests.post('http://localhost:8000/predict/logistic-regression', json=data)
>>> r.json()
{'prediction': False}
```


