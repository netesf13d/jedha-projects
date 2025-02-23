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

region = "eu-west-3"

# =============================================================================
# 
# =============================================================================


def init_temp_file(fname: str, last: datetime)-> datetime:    
    """
    Initialize file for temporary data storage.
    """
    try:
        with open(fname, 'rt', encoding='utf-8') as f:
            for line in f: # go to last line
                pass
            try:
                print(line)
                last_ = datetime.fromisoformat(line.split(';', 1)[0])
                last_ = last_.replace(tzinfo=UTC)
            except UnboundLocalError: # file exists but is empty
                last_ = last
    except (FileNotFoundError, OSError):
        open(fname, 'ab').close()
        last_ = last
    
    return max(last_, last)


def append_to_temp(fname: str, csv_str: str)-> None:
    with open(fname, 'at', encoding='utf-8') as f:
        f.write(csv_str)


def transfer_to_archive(archive_fname: str, temp_fname: str):
    # get last entry timestamp
    # yes yes it would be better to keep it as a metadata
    with open(archive_fname, 'rt', encoding='utf-8') as f:
        for line in f:
            pass
    last = line.split(';', 1)[0]
    # 
    with open(temp_fname, 'rt', encoding='utf-8') as f:
        update = f.readlines()
    # 
    with open(archive_fname, 'at', encoding='utf-8') as f:
        for line in update:
            if line.split(';', 1)[0] > last:
                f.write(line)
        

def update_archive_s3(s3_client,
                      bucket_name: str,
                      archive_fname: str,
                      temp_dir: str):
    # get archive data
    with BytesIO() as f:
        s3_client.download_fileobj(bucket_name, archive_fname, f)
        data = f.getvalue().decode('utf-8')
    df = pd.read_csv(StringIO(data), sep=';')
    
    # get last entry timestamp
    # yes yes it would be better to keep it as a metadata
    with open(archive_fname, 'rt', encoding='utf-8') as f:
        for line in f:
            pass
    last = line.split(';', 1)[0]
    # 
    with open(temp_fname, 'rt', encoding='utf-8') as f:
        update = f.readlines()
    # 
    with open(archive_fname, 'at', encoding='utf-8') as f:
        for line in update:
            if line.split(';', 1)[0] > last:
                f.write(line)



def upload_archive_s3(s3_client,
                      bucket_name: str,
                      archive_fname: str):
    """
    Upload archive to S3 bucket if it does not already exist.
    """
    s3_objs = s3_client.list_objects_v2(Bucket=bucket_name)
    if not bucket_name in {obj['Key'] for obj in s3_objs['Contents']}:
        s3_client.upload_file(bucket_name, Bucket=bucket_name, Key=archive_fname)
        



def update_data(filename: str, last: datetime)-> pd.DataFrame:
    now = datetime.now(tz=UTC)
    
    #
    data = from_forecast()
    df = pd.DataFrame(data['hourly']).set_index('time')
    df = df.loc[df.index < now.isoformat()]
    
    if (now - last).days >= 2:
        data = from_archive(now.date().isoformat(), now.date.isoformat())
        df_ = pd.DataFrame(data['hourly']).setindex('time')
        df_ = df_.dropna(how='all', axis=0)
        df = df.update(df_)
    
    return df.sort_index()


