# -*- coding: utf-8 -*-
"""
Thi

https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
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



# =============================================================================
# 
# =============================================================================


def save_file(s3_client, ):
    pass


def download_file(s3_client, filename: str)-> str:
    with BytesIO() as f:
        s3_client.download_fileobj(bucket_name, filename, f)
        data = f.getvalue().decode('utf-8')
    return data


# =============================================================================
# 
# =============================================================================

region_name = "eu-west-3"
bucket_name = "jedha-project-1-01"
with open("../bucket_name.key", 'rt', encoding='utf-8') as f:
    bucket_name = f.read()
with open("../jedha-project-s3-writer_accessKeys.key", 'rt', encoding='utf-8') as f:
    aws_access_key_id, aws_secret_access_key = f.readlines()[-1].strip().split(',')


s3 = boto3.client('s3', region_name=region_name,
                  aws_access_key_id=aws_access_key_id, 
                  aws_secret_access_key=aws_secret_access_key)

# s3.upload_file('./data/last_data.csv', Bucket=bucket_name, Key='test/super_data.csv')

s3_objs = s3.list_objects_v2(Bucket=bucket_name)


sys.exit()

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






