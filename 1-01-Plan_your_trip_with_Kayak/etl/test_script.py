# -*- coding: utf-8 -*-
"""
Thi

https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

requires psycopg[binary,pool]
"""

import sys
import time
from datetime import datetime, UTC, tzinfo
from io import BytesIO, StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd
import requests
from requests import RequestException

from api_mgmt import fetch_update

import psycopg2

conn = psycopg2.connect(
    dbname='dbname',
    user='alex',
    password='AbC123dEf',
    host='ep-cool-darkness-123456.us-east-2.aws.neon.tech',
    port='5432',
    sslmode='verify-full',
    sslrootcert='/path/to/your/root.crt'
)






sys.exit()


# script part

with open("bucket_name.txt", 'rt', encoding='utf-8') as f:
    bucket_name = f.read()
with open("AWS_S3_cred.key", 'rt', encoding='utf-8') as f:
    aws_access_key_id, aws_secret_access_key = f.readlines()[-1].split(',')



s3 = boto3.client('s3',
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key, 
)

# s3.upload_file('./data/last_data.csv', Bucket=bucket_name, Key='test/super_data.csv')

s3_objs = s3.list_objects_v2(Bucket=bucket_name)


# with open('data/last_data.csv', 'rb') as f:
#     d = f.read()
# s3.put_object(Bucket=bucket_name, Key='test/super_data.csv',
#               Body=d, WriteOffsetBytes=9)

obj = s3.get_object(Bucket=bucket_name, Key='test/super_data.csv')

with BytesIO() as f:
    s3.download_fileobj(bucket_name, 'test/super_data.csv', f)
    data = f.getvalue().decode('utf-8')


if __name__ == '__main__':
    
    archive_fname = 'data/MARSEILLE 2000-2025.csv'
    temp_fname = 'data/last_data.csv'
    last = datetime.fromisoformat('2025-02-04T18:00Z')
    # last = init_temp_file(temp_fname, last)
    # df = fetch_update(last)
    # append_to_temp(fname, df.to_csv(sep=';'))
    # transfer_to_archive(archive_fname, temp_fname)



