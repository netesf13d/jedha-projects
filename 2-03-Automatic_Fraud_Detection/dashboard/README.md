---
title: Dashboard 2 03 Fraud Detection
emoji: 🕵️‍♀️
sdk: docker
app_port: 8501
tags:
- streamlit
---
# Automatic Fraud Detection Dashboard






## Deployment

- Create a read replica of the transaction database (see [here](https://neon.com/docs/guides/read-only-access-read-replicas) for instance)
- Get a connection to the read replica

### Local deployment

```console
$ export $READ_REPLICA_CONN="<READ_REPLICA_CONNECTION_URI>"
$ streamlit run app.py --server.address localhost --server.port 5000
```

```powershell
$ $env:READ_REPLICA_CONN="<READ_REPLICA_CONNECTION_URI>"
$ streamlit run app.py --server.address localhost --server.port 5000
```

### Deployment on Huggingface



