---
title: Getaround Pricing API
emoji: 🚗
sdk: docker
app_port: 7860
---
# Getaround Pricing API

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


### Deployment on Huggingface

It is assumed that the deployment OS is Linux.
```bash
gunicorn app:app --host localhost --port 8000 --worker-class uvicorn_worker.UvicornWorker
```



## Usage


```bash
curl -X 'POST' \
  'http://localhost:8000/predict/ridge-regression' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_key": "Audi",
  "mileage": 132979,
  "engine_power": 112,
  "fuel": "diesel",
  "paint_color": "brown",
  "car_type": "estate",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": true,
  "winter_tires": true
}'
```


```python
>>> import requests
>>> model_name = 'ridge_regression'
>>> data = {'model_key': 'Audi',
...         'mileage': 132979,
...         'engine_power': 112,
...         'fuel': 'diesel',
...         'paint_color': 'brown',
...         'car_type': 'estate',
...         'private_parking_available': True,
...         'has_gps': True,
...         'has_air_conditioning': False,
...         'automatic_car': False,
...         'has_getaround_connect': True,
...         'has_speed_regulator': True,
...         'winter_tires': True}
>>> r = requests.post('http://localhost:8000/predict/ridge-regression', json=data)
>>> r.json()
{'prediction': 120.81970398796342}
```


