# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from engine_core import Customer, Merchant
from engine_core import (probe_transaction_api, get_transaction,
                         probe_fraud_detection_api, detect_fraud)


CUSTOMER_COLS = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
MERCHANT_COLS = ['merchant']

os.environ['TRANSACTIONS_API_URI'] = 'https://charlestng-real-time-fraud-detection.hf.space/'
os.environ['FRAUD_DETECTION_API_URI'] = 'https://netesf13d-api-2-03-fraud-detection.hf.space/'


# =============================================================================
# Process transactions
# =============================================================================

def connect_to_database():
    """
    !!!
    """
    from sqlalchemy import create_engine
    from engine_core import Base

    engine = create_engine('postgresql://airflow-backend_owner:npg_HhvnZOU6utI2@ep-delicate-bar-a2mvyxfz-pooler.eu-central-1.aws.neon.tech/airflow-backend?sslmode=require', echo=False)
    Base.metadata.create_all(engine)
    return engine


def close_engine(engine):
    """
    Close the engine gracefully.
    """
    engine.dispose()


engine = connect_to_database()

with engine.begin() as conn:
    a = conn.execute(text('SELECT MAX(merchant_id) FROM merchants'))
    
    
# trans = conn.begin()
# try:
#     conn.execute()
# except Exception:
#     trans.rollback()
#     raise



# transaction = get_transaction(os.environ['TRANSACTIONS_API_URI'])
# transaction = dict(pd.DataFrame(**transaction).iloc[0])

# merch_filt = {col: transaction[col] for col in MERCHANT_COLS}
# merch_filt['merchant'] = merch_filt['merchant'].strip('fraud_')

# statement = (
#     select(Merchant.merchant_id, Merchant.merch_fraud_victim)
#     .filter_by(**merch_filt)
# )
# with Session(engine) as session:
#     merch_features = session.execute(statement).all()[0]



