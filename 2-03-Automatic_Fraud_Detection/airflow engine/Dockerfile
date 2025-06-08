## Use an official Airflow image as base
FROM apache/airflow:3.0.1

## Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__API__WORKER_TIMEOUT=300
ENV AIRFLOW__API__PORT=7860
ENV AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE="${AIRFLOW_HOME}/simple_auth_manager_passwords.json"
#ENV AIRFLOW__CORE__AUTH_MANAGER=airflow.auth.managers.fab.auth_manager.FabAuthManager

## Switch to root
#USER root

## Copy credentials of the form '{"admin":"<password>"}' to the auth manager file
#RUN --mount=type=secret,id=ADMIN_CREDENTIALS,mode=0444,required=true \
#    cat /run/secrets/ADMIN_CREDENTIALS > "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"

## Copy DAGs and engine core
#COPY --chown=airflow ./dags "$AIRFLOW_HOME/"
#COPY --chown=airflow ./simple_auth_manager_passwords.json "${AIRFLOW_HOME}/"
## Install dependencies
#COPY --chown=airflow requirements.txt .
#RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} -r requirements.txt

# RUN chown airflow "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"


## Change the UID of airflow user to 1000
## Switch back to airflow user (the user is defined in the docker image)
USER root
RUN usermod -u 1000 airflow
USER airflow


## Copy DAGs and engine core
COPY ./dags "$AIRFLOW_HOME/"
COPY ./simple_auth_manager_passwords.json "${AIRFLOW_HOME}/"
## Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} -r requirements.txt


## Install dependencies
#COPY requirements.txt .
#RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
    #apache-airflow-providers-fab \

# !!! test
RUN ls /opt/airflow -la

## Use custom backend database for Airflow. Defaults to local SQLite database.
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$(cat /run/secrets/AIRFLOW_BACKEND_URI)

## Clean and reset the backend database
# RUN airflow db reset
RUN airflow db migrate


# Create default admin user for Airflow (username: admin, password: admin)
# auth_manager = airflow.providers.fab.auth_manager.FabAuthManager
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    airflow users create \
#    --username admin \
#    --firstname Admin \
#    --lastname User \
#    --role Admin \
#    --email admin@example.com \
#    --password $(cat /run/secrets/ADMIN_PASSWORD)

## Start Airflow webserver and scheduler within the same container
CMD ["bash", "-c", "airflow scheduler & airflow api-server"]
