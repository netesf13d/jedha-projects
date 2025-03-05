# Getaround Analysis

In this project we deploy a small application.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `Walmart_sales.ipynb` presents the results or our study. This is the main deliverable for this project.
- The directory `scripts` contains the study as a python script.
- The directory `data` contains the relevant data for the project.
- The file `requirements.txt` gives the list of the project dependencies. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats.


## Usage

To setup the Python environment:
- With `pip`, run `pip install -r requirements.txt`
- Using `conda`, run `conda create --name <env_name> --file requirements.txt`

To run the code, some resources must be created:
- A [S3 bucket](https://aws.amazon.com/s3/) as a data lake to store the collected data. In the notebook `data_collection.ipynb`, the variable `BUCKET_NAME` must be set to the path to a file containing the bucket name.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3FullAccess` policy attached. This profile is used to transfer the data to the S3 bucket. In the notebook `data_collection.ipynb`, the variable `S3_WRITER_ACCESS_KEYS` must be set to the path of the access keys file.
- An [AWS](https://aws.amazon.com/) user with `AmazonS3ReadOnlyAccess` policy attached. This profile is used to transfer the data to the S3 bucket. In the notebook `data_collection.ipynb`, the variable `S3_READER_ACCESS_KEYS` must be set to the path of the access keys file.
- A [Neon](https://neon.tech) database. In both notebooks `data_collection.ipynb` and `Plan_your_trip_with_Kayak.ipynb`, the variable `NEONDB_ACCESS_KEYS` must be set to the path to a file containing the database connection parameters (host, database, user, password).
It is recommended to change the file extention of the credentials to `.key` as such files are ignored by Git int this repository.

Finally, the script `data/make_places_csv.py` must be executed to generate the initial csv file containing the places of interest. This is the starting point of the `data_collection.ipynb`notebook.