# -*- coding: utf-8 -*-
"""

"""

import json

import pandas as pd


# =============================================================================
# 
# =============================================================================

def create_merchants_df()-> pd.DataFrame:
    
    ## discard the 'fraud_' string before each merchant value
    raw_df['merchant'] = raw_df['merchant'].apply(lambda x: x.strip('fraud_'))
    
    ## `merchants` reference table
    merchant_cols = ['merchant']
    merchants_df = raw_df[merchant_cols].value_counts().index.to_frame()
    merchants_df.index = pd.Index(np.arange(1, len(merchants_df)+1),
                                  name='merchant_id')
    
    ## compute merchant victim status
    merch_victim = pd.merge(raw_df[merchant_cols + ['is_fraud']],
                            merchants_df.assign(merchant_id=merchants_df.index),
                            how='left', on=merchant_cols)
    merch_victim = merch_victim[['is_fraud', 'merchant_id']] \
    
    ## add fraudster status to merchants table
    merch_fraud_victim = merch_victim.groupby('merchant_id').any() \
                                     .rename({'is_fraud': 'merch_fraud_victim'}, axis=1)
    merchants_df = merchants_df.join(merch_fraud_victim, validate='one_to_one')
    return merchants_df


def create_customers_df()-> pd.DataFrame:
    
    ## `customer` reference table
    customer_cols = ['cc_num', 'first', 'last', 'gender', 'street', 'city',
                     'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']
    customers_df = raw_df[customer_cols].value_counts().index.to_frame()
    customers_df.index = pd.Index(np.arange(1, len(customers_df)+1),
                                  name='customer_id')
    
    ## compute customer fraudster status
    cust_fraud = pd.merge(raw_df[customer_cols + ['is_fraud']],
                          customers_df.assign(customer_id=customers_df.index),
                          how='left', on=customer_cols)
    cust_fraud = cust_fraud[['is_fraud', 'customer_id']]
    
    ## add fraudster status to customers table
    cust_fraudster = cust_fraud.groupby('customer_id').any() \
                               .rename({'is_fraud': 'cust_fraudster'}, axis=1)
    customers_df = customers_df.join(cust_fraudster, validate='one_to_one')


# =============================================================================
# 
# =============================================================================

def process_transaction():
    pass


