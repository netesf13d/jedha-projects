# Plan your trip with Kayak

This project consists in the contruction of a small application displaying weather information and available hotels in some selected places. It revolves mainly around the Extract-Transform-Load process, and applies the standard techniques used for ETL:
- API calls for the collection of geographic coordinates and weather information;
- Web scraping to get information on available hotels;
- Data transfer to a data lake;
- Data transfer to an SQL database.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `Plan_your_trip_with_Kayak.ipynb` presents the application.
- The Python package `etl` provides the functionality for data collection and storage.
- The Jupyter notebook `data_collection.ipynb` illustrates the usage of the `etl` package to fetch and store the data relevant to the project.
- The directory `data` contains the relevant data for the project.
- The file `requirements.txt` gives the list of the project dependencies. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats.


## Notes

Here is the list of the technologies used throughout this project:
- API calls are done using python [`requests`](https://requests.readthedocs.io/en/latest/) package.
- Web scaping is done using [Selenium](https://www.selenium.dev/) WebDriver.
- The SQL database is provided by [Neon](https://console.neon.tech), with PostgreSQL as the nanagement system. The database is driven with [`psycopg`](https://www.psycopg.org/).
- The data lake provider is [AWS](https://aws.amazon.com/s3/), driven with [`boto3`](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)


- OpenStreetMap