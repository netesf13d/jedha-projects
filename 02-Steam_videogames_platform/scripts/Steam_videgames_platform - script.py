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

"""
The owner ranges are specified on a logarithmic scale (each successive range increases in size by a factor $\sim 2$).
In order to preserve such scaling, we estimate the number of owners from the two bounds by taking their *geometric* mean.
This is problematic for the lowest range (0 - 20000), which we estimate by $\sqrt{20000} \simeq 141}.
We must keep in mind that this last value probably underrates the actual number of owners.
"""

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

* Game pt of view
* Publisher pt of view


OK - Users distrib
OK - prizes distrib
OK - games revenues

- releases evolution in time
- most represented languages
- required_age stats

"""

"""
### Game usage and revenues

We begin by analyzing global game consumption and associated revenues

OK - Users distrib
OK - prizes distrib
OK - games revenues
"""

##
df_ = main_df.loc[:, ['name', 'owners_est', 'price']]
df_ = df_.assign(revenues_est=df_['price']*df_['owners_est'])
print(df_.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99, 0.999]))

##
df_.sort_values('owners_est', ascending=False).iloc[:20]

##
df_.sort_values('revenues_est', ascending=False).iloc[:20]


# TODO fig
##
owners_distrib = main_df['owners_est'].value_counts()

fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
fig1.suptitle('Figure 1: Distribution of game owners', x=0.02, ha='left')

ax1.plot(owners_distrib.index, owners_distrib.to_numpy(), marker='o', linestyle='')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(visible=True)
ax1.set_xlabel('number of owners')
ax1.set_ylabel('number of games')

plt.show()

##
# fig2, axs2 = plt.subplots(
#     nrows=1, ncols=1, figsize=(5, 4), dpi=100,
#     gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
# fig2.suptitle('Figure 1: Distribution of game owners', x=0.02, ha='left')

# axs2.plot(owners_distrib.index, owners_distrib.to_numpy(), marker='o', linestyle='')
# axs2.set_xscale('log')
# axs2.set_yscale('log')
# axs2.grid(visible=True)
# axs2.set_xlabel('number of owners')
# axs2.set_ylabel('number of games')

# plt.show()


"""
The statistics presented above and the distribution of game owners, prices and revenues shown in figure 1
reveal some important characteristics of the game market.
- The distribution of game owners (and therefore of revenues) is fat tailed.
Above a certain threshold, the number of games with a $n$ owners is roughly proportional to $p(n) = \frac{A}{n^{\alpha}},
where the exponent $\alpha$ is an empirical parameter of the distribution. The most important consequence is that a very small
fraction of the games concentrates most of the game usage.
- Some games are free, with sometimes a huge number of owners (eg Dota 2 with more than 200M users). Their apparent revenue (as we estimate it)
is zero. Such games actually generate a major part of their revenue by selling in-games goods on dedicated marketplaces. This is an important
aspect of the video game industry that we will not be able to analyze here.
- The distribution of revenues shows that a large fraction is generated from a minor fraction of the games. Although we are missing 
the revenues from microtransactions, advertisement, etc, the market structure is actually well reproduced by the domination of a few superproductions.
"""

# %%

"""
### Games publishers and developers

We now study how the market is shared among game publishers and developers.

OK - publisher with most releases
"""

##
publishers = main_df['publisher'].value_counts()
publishers

##
developers = main_df['developer'].value_counts()
developers

"""
The total number of publishers/developers is about 30000.
Among these, only 1000 published/developed 6 games or more.
"""

bins = np.logspace(0, 3, 13)
vals = np.sqrt(bins[:-1] * bins[1:])
vals[0] = 1


# fig3, axs3 = plt.subplots(
#     nrows=1, ncols=2, figsize=(7, 4), dpi=100,
#     gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
# fig3.suptitle('Figure 3: Distribution of published/developed games', x=0.02, ha='left')

# axs3[0].plot(vals, np.histogram(publishers, bins)[0], marker='o', linestyle='')
# axs3[0].set_xscale('log')
# axs3[0].set_yscale('log')
# axs3[0].grid(visible=True)
# axs3[0].set_xlabel('number of games published')
# axs3[0].set_ylabel('number of publishers')

# axs3[1].plot(vals, np.histogram(developers, bins)[0], marker='o', linestyle='')
# axs3[1].set_xscale('log')
# axs3[1].set_yscale('log')
# axs3[1].grid(visible=True)
# axs3[1].set_xlabel('number of games developed')
# axs3[1].set_ylabel('number of developers')

# plt.show()

"""
Figure 3 shows the 
"""

##
publishers_by_owners = main_df.loc[:, ['publisher', 'owners_est']] \
                              .groupby('publisher') \
                              .sum() \
                              .sort_values('owners_est', ascending=False)
publishers_by_owners

#
developers_by_owners = main_df.loc[:, ['developer', 'owners_est']] \
                              .groupby('developer') \
                              .sum() \
                              .sort_values('owners_est', ascending=False)
developers_by_owners


## plot

bins = np.concatenate([[1, 10000], np.logspace(4, 9, 16)])
vals = np.sqrt(bins[:-1] * bins[1:])


fig4, axs4 = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
fig4.suptitle('Figure 4: Distribution of published/developed games', x=0.02, ha='left')

axs4[0].plot(vals, np.histogram(publishers_by_owners, bins)[0], marker='o', linestyle='')
axs4[0].set_xscale('log')
axs4[0].set_yscale('log')
axs4[0].grid(visible=True)
axs4[0].set_xlabel('number of games published')
axs4[0].set_ylabel('number of publishers')

axs4[1].plot(vals, np.histogram(developers_by_owners, bins)[0], marker='o', linestyle='')
axs4[1].set_xscale('log')
axs4[1].set_yscale('log')
axs4[1].grid(visible=True)
axs4[1].set_xlabel('number of games developed')
axs4[1].set_ylabel('number of developers')

plt.show()

"""
Figure 4 shows the 
"""

# %% 

"""
### Games release evolution


- releases evolution in time
"""




# %%
"""




- most represented languages
- required_age stats


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





