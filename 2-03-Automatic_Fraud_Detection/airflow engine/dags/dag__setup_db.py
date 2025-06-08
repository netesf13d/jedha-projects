# -*- coding: utf-8 -*-
"""
!!!
Airflow DAGs
Connect to / create database
"""

import logging

from airflow.sdk import DAG, Variable, task



@task(task_id='setup_database')
def setup_database(
        merchants_table_file: str = './merchants_table.csv',
        customers_table_file: str = './customers_table.csv'):
    """
    Establish a connection to the database and create the tables if they are
    not already present. The database contains 3 tables:
    - `transactions`
    - `merchants`
    - `customers`
    """
    import pandas as pd
    from sqlalchemy.orm import Session
    from sqlalchemy import create_engine
    from engine_core import Base, Merchant, Customer

    engine = create_engine(Variable.get('AIRFLOW_CONN_TRANSACTION_DB'),
                           echo=False)
    Base.metadata.create_all(engine)

    ## Add merchants and customers table content to database if not present
    with Session(engine) as session:
        n_merchants = session.query(Merchant).count()
        n_customers = session.query(Customer).count()

    ## merchants table absent from database
    if n_merchants < 693:
        logging.info('Setting up `merchants` table...')
        merchants_df = pd.read_csv(merchants_table_file)
        merchants = [Merchant(**dict(row[1]))
                     for row in merchants_df.iterrows()]
        with Session(engine) as session:
            session.add_all(merchants)
            session.commit()
    logging.info('Table `merchants` set')

    ## customers table absent from database
    if n_customers < 924:
        logging.info('Setting up `customers` table...')
        customers_df = pd.read_csv(customers_table_file)
        customers = [Customer(**dict(row[1]))
                     for row in customers_df.iterrows()]
        with Session(engine) as session:
            session.add_all(customers)
            session.commit()
    logging.info('Table `customers` set')

    logging.info('Database setup complete')
    engine.dispose()


with DAG(
    dag_id='setup_database',
    description=('Probe connection with the database, create schema, '
                 'load the reference tables. '
                 'This DAG is to be triggered manually.'),
    max_active_runs=1,
    default_args={'owner': 'admin', 'depends_on_past': False},
    schedule=None,
    catchup=False,
    tags=['database'],
) as dag:
    setup_database()