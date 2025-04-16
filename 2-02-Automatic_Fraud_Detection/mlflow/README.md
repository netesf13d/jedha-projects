---
title: Getaround project pricing model MLFlow server
emoji: 🚗
sdk: docker
app_port: 7860
---

# Getaround Pricing Model Training

T


## Contents

## Setup external ressources

To run the code, some resources must be created:
- A [S3 bucket](https://aws.amazon.com/s3/) as a data lake to store the collected data.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3FullAccess` policy attached.
- A [Neon](https://neon.tech) database.



## MLFlow tracking server deployment

### Local deployment

Create a file `secrets.sh`:
```
export MLFLOW_TRACKING_URI="http://localhost:7860"
export AWS_ACCESS_KEY_ID="AKIA****************"
export AWS_SECRET_ACCESS_KEY="****************************************"
export BACKEND_STORE_URI="{dialect}+{driver}://{user}:{password}@{hostname}:{port}/{database-name}?sslmode=require"
export ARTIFACT_STORE_URI="s3://{bucket-name}""https://mlflow-artifact-store-1-05-getaround.s3.eu-west-3.amazonaws.com"
```

Set the environment variables 
```bash
source secrets.sh
```

Run the MLFlow server
```bash
mlflow server --host 0.0.0.0 --port 7860 --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $ARTIFACT_STORE_URI
```


### Cloud deployment

The tracking server is deployed in an Huggingface space.

- Create a huggingface [space](https://huggingface.co/new-space). Choose `docker` as the software development kit.
- Setup the variable and secret in the space's settings. Here the `$PORT` variable is set to 7860, the default value (in any case, it must match the `app_port` config variable of the space).




```python
## create S3 client with writing permission
with open(BUCKET_NAME, 'rt', encoding='utf-8') as f:
    bucket_name = f.read()
with open(S3_WRITER_ACCESS_KEYS, 'rt', encoding='utf-8') as f:
    aws_access_key_id, aws_secret_access_key = f.readlines()[-1].strip().split(',')

s3_writer = boto3.client('s3', # region_name=region_name,
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)

## Uplpoad the files created
s3_writer.upload_file('./data/locations.csv', Bucket=bucket_name, Key='data/locations.csv')
s3_writer.upload_file('./data/weather_indicators.csv', Bucket=bucket_name, Key='data/weather_indicators.csv')
s3_writer.upload_file('./data/hotels.csv', Bucket=bucket_name, Key='data/hotels.csv')
```


## Model training on AWS EC2 instances

- Create an EC2 user and an EC2 instance on AWS. EC2 user has `AmazonEC2FullAccess` permission policy.



## Usage




