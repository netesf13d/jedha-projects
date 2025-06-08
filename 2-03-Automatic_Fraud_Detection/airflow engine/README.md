---
title: Automatic Fraud Detection project Airflow server
emoji: 🌪️
sdk: docker
app_port: 7860
---



## Contents


To [export the DAGs as images](https://airflow.apache.org/docs/apache-airflow/stable/howto/usage-cli.html#exporting-dags-structure-to-images) (here the DAG `process_transaction`):
```bash
airflow dags show process-transaction --save process-transaction.png
```

Connection (management)[https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html]



## Deployment

To run the code, some resources must be created:
- A database to store the transactions (e.g., with [Neon](https://neon.tech)).
- (Optional) A PostgreSQL database as the Airflow server backend store (e.g., with [Neon](https://neon.tech)). Ensure the Postgres version is compatible with the version of Airflow used.


The deployment requires some variables and secrets to be set:
- Secret `AIRFLOW_CONN_TRANSACTION_DB`, (the substring `'TOKEN'` is parsed by Airflow and the corresponding variable is hidden from logs)
- Secret `ADMIN_PASSWORD`,
- Variable `AIRFLOW_VAR_TRANSACTIONS_API_URI`, 
- Variable `AIRFLOW_VAR_FRAUD_DETECTION_API_URI`,
- (Optional) Secret `AIRFLOW_BACKEND_URI`, backend database uri for Airflow.


### Local deployment

To setup the Python environment:
- With `pip`, run `pip install apache-airflow==3 -r requirements.txt`
- Using `conda`, run `conda create --name <env_name> apache-airflow==3 --file requirements.txt`, then switch to environment `conda activate <env_name>`

Reset / initialize the backend database:
```bash
airflow db reset
airflow db migrate
```

Parse DAG files, exclude DAG examples:
```bash
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow dags reserialize
```

Start the scheduler in a console:
```bash
airflow scheduler
```

In another console, set the variables and secrets, start the API server:
```bash
airflow connections add transaction_db --conn-uri <AIRFLOW_CONN_TRANSACTION_DB>
export AIRFLOW_VAR_TRANSACTIONS_API_URI=<AIRFLOW_VAR_TRANSACTIONS_API_URI>
export AIRFLOW_VAR_FRAUD_DETECTION_API_URI=<AIRFLOW_VAR_FRAUD_DETECTION_API_URI>
airflow api-server --port 8080
```


### Deployment on Hugging Face



1. Create a huggingface [space](https://huggingface.co/new-space). Choose `docker` as the software development kit.
2. Transfer the contents of this directory: `dags/`, `Dockerfile`, `README.md`, `requirements.txt`.
3. Setup the variables and secrets in the space's settings [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html)
  - Secret `AIRFLOW_CONN_TRANSACTION_DB`, (the substring `'TOKEN'` is parsed by Airflow and the corresponding variable is hidden from logs)
  - Secret `AIRFLOW_BACKEND_URI`, PostgreSQL database uri for Airflow.
  - Secret `ADMIN_PASSWORD`,
  - Variable `AIRFLOW_VAR_TRANSACTIONS_API_URI`, 
  - Variable `AIRFLOW_VAR_FRAUD_DETECTION_API_URI`,
  - Variable `PORT`. Here the value is set to 7860, the default value (in any case, it must match the `app_port` config variable of the space).
4. Run the space and set it public. It will be accessible at `https://{owner}-{your-space-name}.hf.space/`.