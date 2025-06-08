## Use an official Airflow image as base
FROM apache/airflow:3.0.1

## Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__DAGS_FOLDER="${AIRFLOW_HOME}/dags"
ENV AIRFLOW__API__WORKER_TIMEOUT=300
ENV AIRFLOW__API__PORT=7860
ENV AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE="${AIRFLOW_HOME}/simple_auth_manager_passwords.json"


USER root

## Change the UID of airflow user to 1000
RUN usermod -u 1000 airflow

## Copy credentials of the form '{"admin":"<password>"}' to the auth manager file
#RUN --mount=type=secret,id=ADMIN_CREDENTIALS,mode=0444,required=true \
#    cat /run/secrets/ADMIN_CREDENTIALS > "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"

## Copy DAGs, setup permissions
COPY ./dags $AIRFLOW__CORE__DAGS_FOLDER
RUN chown -R airflow $AIRFLOW__CORE__DAGS_FOLDER
COPY ./simple_auth_manager_passwords.json "${AIRFLOW_HOME}/"
RUN chown airflow "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"

## Switch back to airflow user
USER airflow

## Install dependencies
COPY requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt

# !!! test
RUN ls /opt/airflow -la
RUN ls /opt/airflow/dags -la

## Use custom backend database for Airflow. Defaults to local SQLite database.
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$(cat /run/secrets/AIRFLOW_BACKEND_URI)

## Setup the backend database
# RUN airflow db reset -y
RUN airflow db migrate

## Fetch DAGs from files
RUN airflow dags reserialize

## Transfer secret connections
USER root
RUN --mount=type=secret,id=AIRFLOW_CONN_TRANSACTION_DB,mode=0444,required=true \
    airflow connections add transaction_db --conn-uri $(cat /run/secrets/AIRFLOW_CONN_TRANSACTION_DB)
USER airflow

## Start Airflow webserver and scheduler within the same container
CMD ["bash", "-c", "airflow scheduler & airflow api-server"]
