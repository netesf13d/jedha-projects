# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
"""

from datetime import datetime, timedelta
import os

from airflow.models import Variable
from airflow.sdk import DAG, task


# =============================================================================
# Shared tasks
# =============================================================================

@task(task_id='connect_to_db')
def connect_to_database():
    """
    !!!
    """
    from sqlalchemy import create_engine
    from core import Base
    
    engine = create_engine(os.environ['DATABASE_URI'], echo=False)
    Base.metadata.create_all(engine)
    return engine


@task(task_id='close_connection')
def close_engine(engine):
    """
    Close the engine gracefully.
    """
    engine.dispose()


# =============================================================================
# Check environment variables and API servers
# =============================================================================

with DAG(
    dag_id='check-environment',
    description=('Check that all the necessary resources are present. '
                 'This DAG is to be triggered manually in case of problem.'),
    default_args={
        'owner': 'airflow',
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule=None,
    catchup=False,
    tags=['config'],
) as dag:
    
    @task('chk_env_vars')
    def check_env_vars():
        from core import check_environment_vars
        check_environment_vars()
    
    
    @task(task_id='probe_transact_api')
    def probe_transact_api():
        from httpx import HTTPError
        from core import probe_transaction_api
        try:
            probe_transaction_api(os.environ['TRANSACTIONS_API_URI'])
        except HTTPError as e:
            raise HTTPError(f'Transaction API unvailable: {e}')
    
    
    @task(task_id='probe_fraud_api')
    def probe_fraud_api()-> bool:
        from httpx import HTTPError
        from core import probe_fraud_detection_api
        try:
            probe_fraud_detection_api(os.environ['FRAUD_DETECTION_API_URI'])
        except HTTPError:
            Variable.set(key='fraud_api_available', value=False)
        else:
            Variable.set(key='fraud_api_available', value=True)
    
    check_env_vars >> [probe_transact_api, probe_fraud_api]


# =============================================================================
# Connect to / create database
# =============================================================================

with DAG(
    dag_id='database-setup',
    description=('Probe connection with the database, create schema, '
                 'load the reference tables. '
                 'This DAG is to be triggered manually.'),
    default_args={
        'owner': 'airflow',
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule=None,
    catchup=False,
    tags=['database'],
) as dag:
    
    @task(task_id='setup_ref_tables')
    def setup_reference_tables(
            engine,
            merchants_table_file: str = './merchants_table.csv',
            customers_table_file: str = './customers_table.csv'):
        """
        !!!
        """
        import pandas as pd
        from sqlalchemy.orm import Session
        from core import Merchant, Customer
        
        ## Add merchants and customers table content to database if not present
        with Session(engine) as session:
            n_merchants = session.query(Merchant).count()
            n_customers = session.query(Customer).count()
        
        if n_merchants < 693: # merchants table absent from database
            dag.log.info('Setting up `merchants` table...')
            merchants_df = pd.read_csv('./merchants_table.csv')
            merchants = [Merchant(**dict(row[1])) for row in merchants_df.iterrows()]
            with Session(engine) as session:
                session.add_all(merchants)
                session.commit()
        dag.log.info('Table `merchants` set')
        
        if n_customers < 924: # customers table absent from database
            dag.log.info('Setting up `customers` table...')
            customers_df = pd.read_csv('./customers_table.csv')
            customers = [Customer(**dict(row[1])) for row in customers_df.iterrows()]
            with Session(engine) as session:
                session.add_all(customers)
                session.commit()
        dag.log.info('Table `customers` set')
    
    
    _engine = connect_to_database()
    setup_reference_tables(_engine) >> close_engine(_engine)


# =============================================================================
# Process transactions
# =============================================================================

with DAG(
    dag_id='process_transaction',
    description='Setup the database connection',
    default_args={
        'owner': 'airflow',
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    schedule=timedelta(seconds=20),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['pipeline'],
) as dag:
    
    @task(task_id='get_transaction_id')
    def get_transaction_id(engine)-> int:
        """
        Get last transaction id, initialize to 1 if `transactions` table is empty
        """
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session
        from core import Transaction
        
        statement = select(func.max(Transaction.transaction_id))
        with Session(engine) as session:
            transact_ids = session.execute(statement).all()[0][0]
        transaction_id = 1 if transact_ids is None else max(transact_ids) + 1
        return transaction_id


    @task(task_id='get_transaction')
    def fetch_transaction(engine)-> dict:
        """
        !!!
        """
        from httpx import HTTPError
        import pandas as pd
        from core import get_transaction
        try:
            transaction = get_transaction(os.environ['TRANSACTIONS_API_URI'])
        except HTTPError as e:
            print(f"Error fetching transaction: {e}")
        else:
            return dict(pd.DataFrame(**transaction).iloc[0])
    
    
    @task(task_id='get_user_features', multi_outputs=True)
    def get_user_features(engine, transaction: dict)-> dict[str, dict]:
        """
        !!!
        """
        from core import customer_features, merchant_features
        
        return {'merch_features': merchant_features(transaction, engine),
                'cust_features': customer_features(transaction, engine)}
    
    
    @task(task_id='get_fraud_risk')
    def get_fraud_risk(transaction: dict,
                       merch_features: dict,
                       cust_features: dict,
                       fraud_model: str = 'random-forest')-> bool | None:
        """
        !!!
        """
        from core import fraud_detection_features, detect_fraud
        if Variable.get(key='fraud_api_available'):
            pred_features = fraud_detection_features(
                transaction, merch_features, cust_features)
            return detect_fraud(os.environ['FRAUD_DETECTION_API_URI'],
                                fraud_model, pred_features)
    

    @task(task_id='record_transaction')
    def record_transaction(engine,
                           transaction: dict, transaction_id: int,
                           merch_features: dict, cust_features: dict,
                           fraud_risk: bool | None):
        """
        !!!
        """
        from sqlalchemy.orm import Session
        from core import transaction_entry
        
        import numpy as np
        from psycopg2.extensions import register_adapter, AsIs
        register_adapter(np.float64, AsIs)
        register_adapter(np.int64, AsIs)
        
        transaction_out = transaction_entry(transaction, transaction_id,
                                            merch_features, cust_features,
                                            fraud_risk)
        with Session(engine) as session:
            session.add(transaction_out)
            session.commit()
            transaction_id += 1
        dag.log.info(f'Record transaction with id {transaction_id}')


    _engine = connect_to_database
    _transaction_id = get_transaction_id(_engine)
    _transaction = fetch_transaction(_engine)
    _user_features = get_user_features(_engine, _transaction)
    _fraud_risk = get_fraud_risk(_transaction,
                                 _user_features['merch_features'],
                                 _user_features['_cust_features'])
    
    (record_transaction(_engine, _transaction, _transaction_id,
                        _user_features['merch_features'],
                        _user_features['_cust_features'],_fraud_risk)
     >> close_engine(_engine))

