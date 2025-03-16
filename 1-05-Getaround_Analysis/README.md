# Getaround Analysis

In this project we deploy a small application.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `Project_description.ipynb` details the project description.
- The Jupyter notebook `Getaround_Analysis_pricing.ipynb` presents some exploratory data analysis along with two models for car rental pricing optimization.
- The Jupyter notebook `Getaround_Analysis_delay_analysis.ipynb` contains exploratory data analysis of common car rental issues such as late checkout.
- The directory `scripts` contains the two studies as standalone python scripts.
- The directory `data` contains the relevant data for the project.
- The file `requirements.txt` gives the list of notebook dependencies.
- The directory `api` contains !!!
- The directory `dashboard` contains !!!
- The directory `train` contains scripts and utilities to dispatch the training of pricing models on AWS EC2 instances. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats.


## Usage

#### Setup the python environment

- With `pip`, run `pip install -r requirements.txt`
- Using `conda`, run `conda create --name <env_name> --file requirements.txt`


#### Setup external ressources

To run the MLFlow server, some resources must be created:
- A [S3 bucket](https://aws.amazon.com/s3/) as a data lake to store the collected data. In the notebook `data_collection.ipynb`, the variable `BUCKET_NAME` must be set to the path to a file containing the bucket name.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3FullAccess` policy attached. This profile is used to transfer the data to the S3 bucket. In the notebook `data_collection.ipynb`, the variable `S3_WRITER_ACCESS_KEYS` must be set to the path of the access keys file.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3ReadOnlyAccess` policy attached. This profile is used to transfer the data to the S3 bucket. In the notebook `data_collection.ipynb`, the variable `S3_READER_ACCESS_KEYS` must be set to the path of the access keys file.
- A [Neon](https://neon.tech) database. In both notebooks `data_collection.ipynb` and `Plan_your_trip_with_Kayak.ipynb`, the variable `NEONDB_ACCESS_KEYS` must be set to the path to a file containing the database connection parameters (host, database, user, password).
It is recommended to change the file extention of the credentials to `.key` as such files are ignored by Git in this repository.


#### Deploy the MLFlow server


#### Deploy the API


#### Deploy the dashboard

