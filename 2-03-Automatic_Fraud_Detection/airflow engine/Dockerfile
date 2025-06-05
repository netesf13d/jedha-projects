# Use an official Airflow image as base
FROM apache/airflow:3.0.1

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__API__WORKER_TIMEOUT=300
ENV AIRFLOW__API__PORT=7860
# ENV AIRFLOW__CORE__AUTH_MANAGER = airflow.providers.fab.auth_manager.FabAuthManager

# Switch user
USER root

# Copy DAGs and engine core
COPY ./engine_core $AIRFLOW_HOME/dags/engine_core
COPY ./airflow_dags.py $AIRFLOW_HOME/dags/

# Change the UID of airflow user to 1000
RUN usermod -u 1000 airflow

# Switch back to airflow user
USER airflow

# Install any additional dependencies if needed
COPY requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} \
    apache-airflow-providers-fab \
    -r requirements.txt


# !!! test
RUN ls /opt/airflow -la


# Initialize the Airflow database (PostgreSQL in this case)
# or use default local sqlite database 
#RUN --mount=type=secret,id=AIRFLOW_BACKEND_URI,mode=0444,required=true \
#    export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$(cat /run/secrets/AIRFLOW_BACKEND_URI)
# ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$AIRFLOW_BACKEND_URI

RUN echo $AIRFLOW__API__WORKER_TIMEOUT
RUN echo $AIRFLOW_HOME

# Clean and reset the backend database
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

# Start Airflow webserver and scheduler within the same container
CMD ["bash", "-c", "airflow scheduler & airflow api-server"]
