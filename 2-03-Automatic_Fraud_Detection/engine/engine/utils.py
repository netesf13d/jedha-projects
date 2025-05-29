# -*- coding: utf-8 -*-
"""
Utilities
"""

import os


def check_environment_vars()-> None:
    """
    Check that the following environment variables are set in order to
    interact with the database and APIs:
    - `DATABASE_URI`: the URI of the postgreSQL database
    - `TRANSACTIONS_API_URI`: the url of the API giving new transactions
    - `FRAUD_DETECTION_API_URI`: the url of the fraud detection API
    """
    if 'DATABASE_URI' not in os.environ:
        raise KeyError('environment variable `DATABASE_URI` is not set')
    if 'TRANSACTIONS_API_URI' not in os.environ:
        raise KeyError('transactions API url environment variable '
                       '`TRANSACTIONS_API_URI` is not set')
    if 'FRAUD_DETECTION_API_URI' not in os.environ:
        raise KeyError('fraud detection API url environment variable '
                       ' `FRAUD_DETECTION_API_URI` is not set')







