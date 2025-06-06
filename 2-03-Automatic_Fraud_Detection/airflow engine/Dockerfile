## Use an official Airflow image as base
FROM apache/airflow:3.0.1

## Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__API__WORKER_TIMEOUT=300
ENV AIRFLOW__API__PORT=7860
ENV AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE = "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"
# ENV AIRFLOW__CORE__AUTH_MANAGER = airflow.providers.fab.auth_manager.FabAuthManager

## Set engine environment variables (not necessary on Hugging face)
ENV AIRFLOW_VAR_TRANSACTIONS_API_URI=$AIRFLOW_VAR_TRANSACTIONS_API_URI
ENV AIRFLOW_VAR_FRAUD_DETECTION_API_URI=$AIRFLOW_VAR_TRANSACTIONS_API_URI
ENV AIRFLOW_VAR_DATABASE_URI=$AIRFLOW_VAR_DATABASE_URI

## Switch to root
USER root

## Copy credentials of the form '{"admin":"<password>"}' to the auth manager file
RUN --mount=type=secret,id=ADMIN_CREDENTIALS,mode=0444,required=true \
    $(cat /run/secrets/AIRFLOW_BACKEND_URI) > "${AIRFLOW_HOME}/simple_auth_manager_passwords.json"
RUN echo $(cat "${AIRFLOW_HOME}/simple_auth_manager_passwords.json")

## Copy DAGs and engine core
COPY ./engine_core $AIRFLOW_HOME/dags/engine_core
COPY ./airflow_dags.py $AIRFLOW_HOME/dags/

## Change the UID of airflow user to 1000
RUN usermod -u 1000 airflow

## Switch back to airflow user
USER airflow

## Install dependencies
COPY requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} \
    apache-airflow-providers-fab \
    -r requirements.txt


# !!! test
RUN ls /opt/airflow -la


## Use custom backend database for Airflow. Defaults to local SQLite database.
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$(cat /run/secrets/AIRFLOW_BACKEND_URI)
# ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$AIRFLOW_BACKEND_URI

RUN echo $AIRFLOW__API__WORKER_TIMEOUT
RUN echo $AIRFLOW_HOME

# RUN airflow config list --defaults > "${AIRFLOW_HOME}/airflow.cfg"



## Clean and reset the backend database
# RUN airflow db reset
RUN airflow db migrate



# Create default admin user for Airflow (username: admin, password: admin)
# auth_manager = airflow.providers.fab.auth_manager.FabAuthManager
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#RUN airflow users create \
#    --username admin \
#    --firstname Admin \
#    --lastname User \
#    --role Admin \
#    --email admin@example.com \
#    --password test
# $(cat /run/secrets/ADMIN_PASSWORD)

## Start Airflow webserver and scheduler within the same container
CMD ["bash", "-c", "airflow scheduler & airflow api-server"]
