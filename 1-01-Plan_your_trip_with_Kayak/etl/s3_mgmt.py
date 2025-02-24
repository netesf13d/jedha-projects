# -*- coding: utf-8 -*-
"""
Functions to interact with S3 buckets.
"""

import csv
from csv import QUOTE_STRINGS
from io import BytesIO, StringIO



def download_csv(s3_client,
                 bucket_name: str,
                 filename: str,
                 delimiter: str = ';')-> list[dict[str, str]]:
    """
    Download a CSV file from an S3 bucket and return the contents as a list
    of dicts.
    """
    with BytesIO() as f:
        s3_client.download_fileobj(bucket_name, filename, f)
        raw_data = f.getvalue().decode('utf-8')
    with StringIO(raw_data, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter, quoting=QUOTE_STRINGS)
        data = [row for row in reader]
    return data