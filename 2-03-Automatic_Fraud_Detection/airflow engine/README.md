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


## Setup external ressources

To run the code, some resources must be created:
- A database to store the transactions (e.g., with [Neon](https://neon.tech)).
- A PostgreSQL database as the Airflow server backend store (e.g., with [Neon](https://neon.tech)). Ensure the Postgres version is compatible with the version of Airflow used.



## Deployment

### Local deployment

Set the environment variables:
```
export MLFLOW_TRACKING_URI="http://localhost:7860"
export AWS_ACCESS_KEY_ID="AKIA****************"
export AWS_SECRET_ACCESS_KEY="****************************************"
export BACKEND_STORE_URI="{dialect}+{driver}://{user}:{password}@{hostname}:{port}/{database-name}?sslmode=require"
export ARTIFACT_STORE_URI="s3://{bucket-name}"
```

Run the MLFlow server
```bash
mlflow server --host 0.0.0.0 --port 7860 --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $ARTIFACT_STORE_URI
```


### Cloud deployment

The tracking server is deployed in an Huggingface space.

1. Create a huggingface [space](https://huggingface.co/new-space). Choose `docker` as the software development kit.
2. Transfer the contents of this directory: `Dockerfile`, `README.md`, `requirements.txt`.
3. Setup the variables and secrets in the space's settings [here](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html)
  - Secret `AIRFLOW_VAR_DATABASE_TOKEN_URI`, (the substring `'TOKEN'` is parsed by Airflow and the corresponding variable is hidden from logs)
  - Secret `AIRFLOW_BACKEND_URI`, PostgreSQL database uri for Airflow.
  - Secret `ADMIN_PASSWORD`,
  - Variable `AIRFLOW_VAR_TRANSACTIONS_API_URI`, 
  - Variable `AIRFLOW_VAR_FRAUD_DETECTION_API_URI`,
  - Variable `PORT`. Here the value is set to 7860, the default value (in any case, it must match the `app_port` config variable of the space).
4. Run the space and set it public. It will be accessible at `https://{owner}-{your-space-name}.hf.space/`.

