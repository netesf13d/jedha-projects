# Build from
FROM continuumio/miniconda3

WORKDIR /home/app

COPY requirements.txt /dependencies/requirements.txt
COPY . /home/app

RUN pip install -r /dependencies/requirements.txt


#RUN conda create --name api_env \
#    --channel conda-forge \
#    python=3.12 \
#    --file /dependencies/requirements.txt \
#    --yes
#RUN conda activate api_env

# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
# ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
# ENV ARTIFACT_STORE_URI=$ARTIFACT_STORE_URI

# Launch the app
CMD gunicorn \
    --bind 0.0.0.0:$PORT \
    --worker-class uvicorn_worker.UvicornWorker \
    app:app