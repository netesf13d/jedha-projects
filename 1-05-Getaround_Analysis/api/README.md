# Getaround Pricing API

In this project we deploy a small application.



## Deployment

### Local deployment

##### Windows

`gunicorn` is not available on Windows, you must use `uvicorn` to run the app:
```shell
uvicorn app:app --reload --host localhost --port 8000
```

##### Linux/Unix

Both `gunicorn` and `uvicorn` are both available. It is recommended to use `uvicorn` (see [here](https://www.uvicorn.org/deployment/)).
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


## Contents

The API provides two endpoints:
- `GET test`: Probe the API for a response
- `POST predict`: Get a car rental price from car attributes


## Usage


```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"input": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]}' http://your-url/predict
```

```python
>>> import requests
>>> r = requests.post("https://your-url/predict", json={
...     "data": [[7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]]
... })
>>> print(response.json())

```


