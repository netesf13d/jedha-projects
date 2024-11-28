#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# import os
# import sys

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# import findspark
# findspark.init("/home/netes/.spark/spark-3.5.3-bin-hadoop3")

# from pyspark.sql import SparkSession
# from pyspark import SparkContext

# sc = SparkContext(appName="jedha_stuff")


# spark = SparkSession.builder.getOrCreate()

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %% Loading
"""
## <a name="loading"></a> Data loading and preprocessing
"""

with open('./steam_game_output.json', 'rt', encoding='utf-8') as f:
    data = json.load(f)

df_dict = {k: [entry['data'][k] for entry in data] for k in data[0]['data']}
df = pd.DataFrame.from_dict(df_dict)

# TODO tags df
# TODO categories df (of bool)
# TODO platforms df
# TODO parse owners
# TODO parse date
# TODO price in cents


# %% EDA
"""
## <a name="eda"></a> Preliminary data analysis

#### 

We begin by getting insights about the store locations.
"""


"""
- sales for each genre
- sales vs tags
- platfor availability
- 


"""


