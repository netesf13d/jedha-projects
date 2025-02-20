# Steam videogames platform

In this project we conduct an exploratory data analysis of the computer videogames ecosystem. We adopt the point of view of a business analyst willing to take data-driven decisions for the development of a new videogame. We study:
- how the market is shared among the sector agents;
- the availability of video games to the public, in terms of age, language and platform;
- the trends in game genres and their evolution.

The dataset is composed from games available in the catalog of Steam's platform. Ths dataset is limited to games released before december 2022 and can be found [here](https://full-stack-bigdata-datasets.s3.amazonaws.com/Big_Data/Project_Steam/steam_game_output.json). An updated dataset (in csv format) can be found [here](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset).


## Structure of the project

This project repository is structured as follows.
- The Jupyter notebook `Steam_videogames_platform.ipynb` presents the results of our study. This is the main deliverable for the certification.
- The directory `scripts` contains the exploratory data analysis as python scripts. There are 2 scripts, one using `pandas` for data analysis, the other using `pyspark`.
- The file `requirements.txt` gives the list of the project dependencies. 
- The directory `presentation` contains the slideshow for the exam, in both `odp` and `pdf` formats. 


## Notes

For the purpose of learning, I tried to rely as much as possible on SQL queries and pyspark API for data analysis. As a consequence, the resulting code is not the most concise and elegant nor the fastest (for such a small dataset).