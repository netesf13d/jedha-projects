# AT&T Spam Detector

This project is an application of deep learning to the treatment of textual data. The goal is to classify SMS messages as hams or spams. We are confronted to sequential data in high dimension. Our approach is tructured as follows:
- Make a simple and interpretable model using engineered features;
- Refine it using statistical approaches
- Refine it further by incorporating the sequential structure of the data

The dataset is a collection of SMS hams and spams from various sources [[1]](#1). The version provided for the project can be found [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). However, this version has many issues related to text encoding and character escaping. This project rather uses the original version[[2]](#1), available [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), which has no such problems.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `AT&T_Spam_Detector.ipynb` presents the results or our study. This is the main deliverable for this project.
- The directory `scripts` contains the study as a python script.
- The directory `data` contains the relevant data for the project.
- The directory `media` condains media files related to the project.
- The file `requirements.txt` gives the list of the project dependencies. 
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



## References

<a id="1">[1]</a> 
Almeida, T., Hidalgo, J. & Yamakami A. (2011). <em>Contributions to the study of SMS spam filtering: new collection and results</em>. DocEng '11. Association for Computing Machinery, New York, NY, USA, 259–262. DOI: [10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742).
<a id="2">[2]</a> 
Almeida, T. & Hidalgo, J. (2011). <em>SMS Spam Collection [Dataset]</em>. UCI Machine Learning Repository. DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).