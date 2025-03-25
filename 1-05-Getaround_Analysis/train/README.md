---
title: Getaround project pricing model training
emoji: \U0001f697
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Getaround Pricing Model Training

T


## Contents



## MLFlow tracking server deployment

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




