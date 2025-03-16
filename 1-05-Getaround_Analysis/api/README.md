# Getaround Pricing API

In this project we deploy a small application.



## Deployment

### Local deployment

```bash
$ gunicorn app:app --bind 0.0.0.0:4000 --worker-class uvicorn.workers.UvicornWorker
```

### Deployment on Huggingface




## Contents

The API provides two endpoints:
- `GET test`: Probe the API for a response
- `POST predict`: Get a car rental price from car attributes


## Usage


```bash
$ curl -i -H "Content-Type: application/json" -X POST -d '{"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}' http://your-url/predict
```

```python
>>> import requests
>>> r = requests.post("https://your-url/predict", json={
...     "data": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]
... })
>>> print(response.json())

```


