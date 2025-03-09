# Conversion Rate Challenge

This project takes the form of a machine learning challenge similar to those proposed on Kaggle. we are given data relative to user interaction with a website, [datascienceweekly.org](https://www.datascienceweekly.org). The goal is to predict whether a user will subscribe to the newsletter based on features such as the age, country of origin, number of pages visited, etc. The dataset is split between a train set for model construction and a test set for model evaluation, with test targets not accessible to the participants during the challenge. The F1-score is the retained metric for model evaluation on this classification task is the F1-score.

The dataset is synthetic. Both train and test sets are provided in this repository to allow for code reproducibility.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `Conversion_rate_challenge.ipynb` presents the results of our study. This is the main deliverable for the certification.
- The directory `data` contains the relevant data for the project.
- The directory `media` condains media displayed in the notbooks.
- The file `requirements.txt` gives the list of the project dependencies. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats.


## Usage

To setup the Python environment:
- With `pip`, run `pip install -r requirements.txt`
- Using `conda`, run `conda create --name <env_name> --file requirements.txt`