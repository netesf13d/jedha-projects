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

with open('../steam_game_output.json', 'rt', encoding='utf-8') as f:
    data = json.load(f)

df_dict = {k: [entry['data'][k] for entry in data] for k in data[0]['data']}
raw_df = pd.DataFrame.from_dict(df_dict)
# parse release date
raw_df = raw_df.assign(
    release_date=pd.to_datetime(raw_df['release_date'], format='mixed'))
# convert price to $
raw_df = raw_df.assign(price=raw_df['price'].astype(float)/100,
                       initialprice=raw_df['initialprice'].astype(float)/100)
# parse nb of owners
# log ranges => geometric mean
owners_range = raw_df['owners'].apply(lambda s: s.replace(',', '').split(' .. '))
owners_range = np.array(owners_range.to_list(), dtype=np.int64)
owners_est = np.prod(np.where(owners_range==0, 1, owners_range), axis=1)
owners_est = np.sqrt(owners_est)

raw_df = raw_df.assign(owners_low=owners_range[:, 0],
                       owners_high=owners_range[:, 1],
                       owners_est=owners_est)


# %% build multiple dataframes

##
main_cols = [
    'appid', 'name', 'developer', 'publisher', 'genre', 'type',
    'positive', 'negative', 'price', 'initialprice', 'discount', 'ccu',
    'release_date', 'required_age', 'owners_low', 'owners_high', 'owners_est'
]
main_df = raw_df.loc[:, main_cols]


##
all_tags = set.union(*[set(tags) for tags in raw_df['tags']])
tags_df = pd.DataFrame.from_dict(
    {'appid': main_df['appid']} |
    {t: [tags.get(t, 0) for tags in raw_df['tags']] for t in all_tags},
    dtype='Sparse[int]')


##
categories = [set(cat) for cat in raw_df['categories']]
all_categories = set.union(*categories)
categories_df = pd.DataFrame.from_dict(
    {'appid': main_df['appid'], 'name': main_df['name']} |
    {c: [(c in cat) for cat in categories] for c in all_categories})


##
platforms_df = pd.DataFrame(raw_df['platforms'].to_list(),
                            index=main_df['appid'])


##
languages = [set(lang.split(', ')) for lang in raw_df['languages']]
all_languages = set.union(*languages)
languages_df = pd.DataFrame.from_dict(
    {'appid': main_df['appid'], 'name': main_df['name']} |
    {lang: [(lang in langs) for langs in languages] for lang in all_languages})


# OK TODO tags df
# OK TODO categories df (of bool)
# OK TODO platforms df
# OK TODO parse owners
# OK TODO parse date
# OK TODO price in cents


# %% Macro
"""
## <a name="macro"></a> Macro-level analysis

- Users distrib
- prizes distrib
- publishers revenues
- publisher with most releases
- releases evolution in time
- most represented languages
- required_age stats

"""

"""
### Users distrib

"""

owners_distrib = main_df['owners_est'].value_counts()


fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(9, 4), dpi=100,
    gridspec_kw={'left': 0.06, 'right': 0.97, 'top': 0.85, 'bottom': 0.13, 'wspace': 0.18})
fig1.suptitle('Figure 1: Age, income and SAT scores distribution', x=0.02, ha='left')

ax1.plot(owners_distrib.index, owners_distrib.to_numpy(), marker='o', linestyle='')
ax1.set_xscale('log')
ax1.set_yscale('log')

plt.show()



"""
- sales for each genre
- sales vs tags
- platform availability
- 


"""

# %% Genres

"""
## <a name="genres"></a> Genres analysis

- most represented in terms of nb games
- best rated genres
- genres by publishers
- number of players by genres
- most lucrative genres (nb players * price)
- genres evolution
"""





# %% Platform

"""
## <a name="platform"></a> Platform analysis


- general platform availability
- genres and platforms
- nb owners and platforms
- price and platform
- platform availability evolution tendency (window release date and look at pllatform availability)
"""


platforms_df.sum()

platforms_df.sum() / len(platforms_df)





