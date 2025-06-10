## Use an official Airflow image as base
FROM apache/airflow:3.0.1

## Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__DAGS_FOLDER="${AIRFLOW_HOME}/dags"
ENV AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager
ENV AIRFLOW__API__WORKER_TIMEOUT=300
ENV AIRFLOW__API__PORT=7860
ENV AIRFLOW__API__BASE_URL=https://netesf13d-airflow-server-2-03-fraud-detection.hf.space/


USER root

## Change the UID of airflow user to 1000
RUN usermod -u 1000 airflow

## Copy DAGs, setup permissions
COPY ./dags $AIRFLOW__CORE__DAGS_FOLDER
RUN chown -R airflow $AIRFLOW__CORE__DAGS_FOLDER

## Switch back to airflow user
USER airflow


## Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} \
    apache-airflow-providers-fab \
    -r requirements.txt

## Setup the backend database and fetch DAGs
# RUN airflow db reset -y
RUN airflow db migrate
RUN airflow dags reserialize


##### Transfer secrets #####
USER root

## External backend database for Airflow. Defaults to local SQLite database.
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$(cat /run/secrets/AIRFLOW_BACKEND_URI)

## Connection to storage database
RUN --mount=type=secret,id=AIRFLOW_CONN_TRANSACTION_DB,mode=0444,required=true \
    airflow connections add transaction_db --conn-uri $(cat /run/secrets/AIRFLOW_CONN_TRANSACTION_DB)

## Create admin account
RUN --mount=type=secret,id=ADMIN_PASSWORD,mode=0444,required=true \
    airflow users create \
    --firstname John --lastname Doe --email a@b.c \
    --role Admin \
    --username admin \
    --password $(cat /run/secrets/ADMIN_PASSWORD)

USER airflow


## Start Airflow webserver and scheduler within the same container
CMD ["bash", "-c", "airflow scheduler & airflow api-server"]
