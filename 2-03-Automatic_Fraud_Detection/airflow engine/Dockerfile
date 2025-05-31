# Use an official Airflow image as base
FROM apache/airflow:3.0.1

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor
ENV AIRFLOW__WEBSERVER__WEB_SERVER_MASTER_TIMEOUT=300
ENV AIRFLOW__WEBSERVER__WORKER_CLASS=gevent
ENV AIRFLOW__WEBSERVER__WEB_SERVER_PORT=$PORT
ENV AWS_DEFAULT_REGION=eu-west-1

# Switch user
USER root

# Copy DAGs and engine core
COPY ./core /opt/airflow/core
COPY ./airflow_dags.py /opt/airflow/

# Change the UID of airflow user to 1000
RUN usermod -u 1000 airflow

# Switch back to airflow user
USER airflow

# Install any additional dependencies if needed
COPY requirements.txt requirements.txt 
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt

# Initialize the Airflow database (PostgreSQL in this case)
# IF YOU WANT TO HAVE THAT RUNNING IN HUGGINGFACE, YOU NEED TO HARD CODE THE VALUE HERE UNFORTUNATELY
# DON'T STAGE THAT IN A PRIVATE REPO BECAUSE THE ENV VARIABLE IS HARD CODED IN PLAIN TEXT
# IF YOU STAGE THAT IN HUGGING FACE SPACE, YOU DON'T HAVE A CHOICE THOUGH
# SO MAKE SURE YOUR SPACE IS PRIVATE
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=$AIRFLOW_BACKEND_URI

# Clean and reset the backend database
RUN airflow db clean --skip-archive
RUN airflow db init

# Create default admin user for Airflow (username: admin, password: admin)
RUN airflow users create \
   --username admin \
   --firstname Admin \
   --lastname User \
   --role Admin \
   --email admin@example.com \
   --password admin $ADMIN_PASSWORD

# Start Airflow webserver and scheduler within the same container
CMD bash \
    -c \
    airflow scheduler \
    & airflow webserver
# CMD ["bash", "-c", "airflow scheduler & airflow webserver"]
