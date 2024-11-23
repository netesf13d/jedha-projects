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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# %% Loading
"""
## <a name="loading"></a> Data loading and preprocessing
"""

df = pd.read_json('./steam_game_output.json')
print(df.iloc[0, 1])



# %% EDA
"""
## <a name="eda"></a> Preliminary data analysis

#### 

We begin by getting insights about the store locations.
"""