# AT&T Spam Detector

This project is an application of deep learning to the treatment of textual data. The goal is to predict the weekly sales of Walmart stores from external factors such as economical indicators or environmental factors. We follow the standart approach to tackle this kind of problem:
- make a preliminary EDA to better understand the data;
- build a simple and easily interpretable model (here a linear regression), and assess its performance;
- try to improve it.

The dataset originally comes from a Kaggle competition organized by Walmart. The original version can be found [here](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data). The dataset used in this project is the same, without the stores information nor the features labeled `MarkDown*`.


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `AT&T_Spam_Detector.ipynb` presents the results or our study. This is the main deliverable for this project.
- The directory `scripts` contains the study as a python script.
- The directory `data` contains the relevant data for the project.
- The file `requirements.txt` gives the list of the project dependencies. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats.