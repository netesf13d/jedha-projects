---
title: Automatic Fraud Detection project MLFlow server
emoji: 💸
sdk: docker
app_port: 7860
---



## Contents



## Setup external ressources

To run the code, some resources must be created:
- A [S3 bucket](https://aws.amazon.com/s3/) as a data lake to store the collected data.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3FullAccess` policy attached.
- A [Neon](https://neon.tech) database.



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
3. Setup the variables and secrets in the space's settings
  - Secret `AWS_ACCESS_KEY_ID`,
  - Secret `AWS_SECRET_ACCESS_KEY`,
  - Secret `BACKEND_STORE_URI`,
  - Secret `ARTIFACT_STORE_URI`,
  - Variable `PORT`. Here the value is set to 7860, the default value (in any case, it must match the `app_port` config variable of the space).
4. Run the space and set it public. It will be accessible at `https://{owner}-{your-space-name}.hf.space/`.

