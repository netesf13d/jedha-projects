#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# import os
import sys

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# import findspark
# findspark.init("/home/netes/.spark/spark-3.5.3-bin-hadoop3")

from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext(appName="jedha_stuff")


spark = SparkSession.builder.getOrCreate()

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.exit()

# %% Loading
"""
## <a name="loading"></a> Data loading and preprocessing
"""

##
with open('../steam_game_output.json', 'rt', encoding='utf-8') as f:
    data = json.load(f)

df_dict = {k: [entry['data'][k] for entry in data] for k in data[0]['data']}
raw_df = pd.DataFrame.from_dict(df_dict)
# ccu means concurrent users
raw_df

##
# parse release date
raw_df = raw_df.assign(
    release_date=pd.to_datetime(raw_df['release_date'], format='mixed'))

# convert price to $
raw_df = raw_df.assign(price=raw_df['price'].astype(float)/100,
                       initialprice=raw_df['initialprice'].astype(float)/100)

# parse required_age as int
required_age = raw_df['required_age'] \
    .astype(str) \
    .apply(lambda s: ''.join(c for c in s if c.isdigit())) \
    .astype(int)
raw_df = raw_df.assign(required_age=required_age)

"""
The owner ranges are specified on a logarithmic scale (each successive range increases in size by a factor $\sim 2$).
In order to preserve such scaling, we estimate the number of owners from the two bounds by taking their *geometric* mean.
This is problematic for the lowest range (0 - 20000), which we estimate by $\sqrt{20000} \simeq 141}.
We must keep in mind that this last value probably underrates the actual number of owners.
"""

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
release_dates = pd.DatetimeIndex(main_df.loc[:, 'release_date'])


##
all_tags = set.union(*[set(tags) for tags in raw_df['tags']])
tags_df = pd.DataFrame.from_dict(
    {t: [tags.get(t, 0) for tags in raw_df['tags']] for t in all_tags},
    dtype='Sparse[int]')


##
genres = [set(genre.split(', ')) for genre in raw_df['genre']]
all_genres = set.union(*genres)
genres_df = pd.DataFrame.from_dict(
    {genre: [(genre in g) for g in genres] for genre in all_genres})


##
categories = [set(cat) for cat in raw_df['categories']]
all_categories = set.union(*categories)
categories_df = pd.DataFrame.from_dict(
    {c: [(c in cat) for cat in categories] for c in all_categories})


##
platforms_df = pd.DataFrame(raw_df['platforms'].to_list())


##
languages = [set(lang.split(', ')) for lang in raw_df['languages']]
all_languages = set.union(*languages)
languages_df = pd.DataFrame.from_dict(
    {lang: [(lang in langs) for langs in languages] for lang in all_languages})


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


##
fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.95, 'top': 0.89, 'bottom': 0.14})
fig1.suptitle('Figure 1: Game prices distribution', x=0.02, ha='left')

ax1.hist(df_['price'], bins = np.linspace(0, 100, 51))
ax1.set_xlim(0, 80)
ax1.set_ylim(0, 20000)
ax1.set_yticks(np.linspace(0, 20000, 11), np.arange(0, 21, 2))
ax1.grid(visible=True)
ax1.set_xlabel('Price')
ax1.set_ylabel('Number of owners (x1000)')

plt.show()

"""
Figure 1 shows the distribution of prices.
There are many free games (7700)
We note the price pattern 4.99, 9.99, 14.99...
"""


# %%
##
owners_distrib = df_['owners_est'].value_counts()

##
tot_owners = df_['owners_est'].sum()
cum_owners = (df_['owners_est'].sort_values().cumsum() / tot_owners).to_numpy()
own_frac = np.linspace([1], [0], 201, endpoint=True)
own_nb_games = len(cum_owners) - np.argmax(cum_owners >= own_frac, axis=-1)
own_nb_games[0] = 0

##
bins = np.concatenate([[0], np.logspace(4, 10, 19)])
revenues_distrib = np.histogram(df_['revenues_est'], bins=bins)[0]
vals = np.sqrt(bins[:-1] * bins[1:])
vals[0] = np.sqrt(1e4)

##
tot_revenues = df_['revenues_est'].sum()
cum_revenues = (df_['revenues_est'].sort_values().cumsum() / tot_revenues).to_numpy()
cum_revenues = cum_revenues[cum_revenues > 0]
rev_frac = np.linspace([1], [0], 201, endpoint=True)
rev_nb_games = len(cum_revenues) - np.argmax(cum_revenues >= rev_frac, axis=-1)
rev_nb_games[0] = 0


##
fig2, axs2 = plt.subplots(
    nrows=2, ncols=2, figsize=(7, 6), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.9, 'bottom': 0.1,
                 'wspace': 0.24, 'hspace': 0.36})
fig2.suptitle('Figure 2: Distribution of game owners and revenues', x=0.02, ha='left')


axs2[0, 0].plot(owners_distrib.index, owners_distrib.to_numpy(),
                marker='o', linestyle='')
axs2[0, 0].set_xscale('log')
axs2[0, 0].set_xlim(1e1, 1e10)
axs2[0, 0].set_yscale('log')
axs2[0, 0].set_ylim(3e-1, 1e5)
axs2[0, 0].grid(visible=True)
axs2[0, 0].set_xlabel('Number of owners')
axs2[0, 0].set_ylabel('Number of games')


axs2[0, 1].plot(own_nb_games, own_frac, marker='', linestyle='-')
axs2[0, 1].plot([0, own_nb_games[100], own_nb_games[100]], [0.5, 0.5, 0],
                color='k', lw=0.8, ls='--')
axs2[0, 1].set_xscale('log')
axs2[0, 1].set_xlim(8e-1, 1e5)
axs2[0, 1].set_ylim(0, 1)
axs2[0, 1].grid(visible=True)
axs2[0, 1].set_xlabel('Number of games')
axs2[0, 1].set_ylabel('Fraction of total game usage')


axs2[1, 0].plot(vals, revenues_distrib, marker='o', linestyle='')
axs2[1, 0].set_xscale('log')
axs2[1, 0].set_xlim(1e1, 1e10)
axs2[1, 0].set_yscale('log')
axs2[1, 0].set_ylim(3e-1, 1e5)
axs2[1, 0].grid(visible=True)
axs2[1, 0].set_xlabel('Game revenue')
axs2[1, 0].set_ylabel('number of games')


axs2[1, 1].plot(rev_nb_games, rev_frac, marker='', linestyle='-')
axs2[1, 1].plot([0, rev_nb_games[100], rev_nb_games[100]], [0.5, 0.5, 0],
                color='k', lw=0.8, ls='--')
axs2[1, 1].set_xscale('log')
axs2[1, 1].set_xlim(8e-1, 1e5)
axs2[1, 1].set_ylim(0, 1)
axs2[1, 1].grid(visible=True)
axs2[1, 1].set_xlabel('Number of games')
axs2[1, 1].set_ylabel('Fraction of total revenues')

plt.show()


"""
The statistics presented above and the distribution of game owners, prices and revenues shown in figure 1
reveal some important characteristics of the game market.
- The distribution of game owners (and therefore of revenues) is fat tailed.
Above a certain threshold, the number of games with a $n$ owners is roughly proportional to $p(n) = \frac{A}{n^{\alpha}},
where the exponent $\alpha$ is an empirical parameter of the distribution. The most important consequence is that a very small
fraction of the games concentrates most of the game usage.
- Most games are free, with sometimes a huge number of owners (eg Dota 2 with more than 200M users). Their apparent revenue (as we estimate it)
is zero. Such games actually generate a major part of their revenue by selling in-games goods on dedicated marketplaces. This is an important
aspect of the video game industry that we will not be able to analyze here.
- The distribution of revenues shows that a large fraction is generated from a minor fraction of the games.
About half the counted revenues originate from 200 games (less than 0.5 %!). Although we are missing 
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

fig3, ax3 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
fig3.suptitle('Figure 3: Distribution of published/developed games', x=0.02, ha='left')

ax3.plot(vals, np.histogram(publishers, bins)[0],
             marker='o', linestyle='', alpha=0.8, label='publishers')
ax3.plot(vals, np.histogram(developers, bins)[0],
             marker='o', linestyle='', alpha=0.7, label='developers')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(visible=True)
ax3.set_xlabel('number of games published')
ax3.set_ylabel('number of publishers')
ax3.legend()

plt.show()

"""
Figure 3 shows the 
"""

##
publishers_by_owners = main_df.loc[:, ['publisher', 'owners_est']] \
                              .groupby('publisher') \
                              .sum() \
                              .sort_values('owners_est', ascending=False)
publishers_by_owners

##
developers_by_owners = main_df.loc[:, ['developer', 'owners_est']] \
                              .groupby('developer') \
                              .sum() \
                              .sort_values('owners_est', ascending=False)
developers_by_owners


## plot
bins = np.concatenate([[1, 10000], np.logspace(4, 9, 16)])
vals = np.sqrt(bins[:-1] * bins[1:])

fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.89, 'bottom': 0.14})
fig4.suptitle('Figure 4: Distribution of published/developed games', x=0.02, ha='left')

ax4.plot(vals, np.histogram(publishers_by_owners, bins)[0],
             marker='o', linestyle='', alpha=0.8, label='publishers')
ax4.plot(vals, np.histogram(developers_by_owners, bins)[0],
             marker='o', linestyle='', alpha=0.7, label='developpers')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(visible=True)
ax4.set_xlabel('Customers base')
ax4.set_ylabel('number of publishers / developers')
ax4.legend()

plt.show()

"""
Figure 4 shows the 
"""

# %% 

"""
### Games release evolution


- releases evolution in time


Comment difficulty to evaluate nb of players evolution
"""

##
release_df = pd.DataFrame({'nb_releases': np.ones(len(release_dates))},
                          index=release_dates)
release_df

##
releases_per_month = release_df.resample('MS').sum()
cumulative_releases = releases_per_month.cumsum()


##
fig5, axs5 = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 3.5), dpi=100,
    gridspec_kw={'left': 0.09, 'right': 0.97, 'top': 0.89, 'bottom': 0.14, 'wspace': 0.24})
fig5.suptitle('Figure 5: Evolution of game releases', x=0.02, ha='left')

axs5[0].plot(releases_per_month.index, releases_per_month,
             marker='', linestyle='-')
axs5[0].set_xlim(12410, 19730)
axs5[0].xaxis.set_major_locator(mdates.YearLocator(4))
axs5[0].set_ylim(0, 900)
axs5[0].grid(visible=True)
axs5[0].set_xlabel('Date')
axs5[0].set_ylabel('Monthly game releases')
axs5[0].legend()


axs5[1].plot(cumulative_releases.index, cumulative_releases,
             marker='', linestyle='-')
axs5[1].set_xlim(12410, 19730)
axs5[1].xaxis.set_major_locator(mdates.YearLocator(4))
axs5[1].set_ylim(0, 60000)
axs5[1].set_yticks(np.linspace(0, 60000, 7), np.arange(0, 70, 10))
axs5[1].grid(visible=True)
axs5[1].set_xlabel('Date')
axs5[1].set_ylabel('Cumulative game releases (x 1000)')

plt.show()

"""
Jump of about 10-20% around covid period
"""



# %%
"""
### Games release evolution



- most represented languages
- required_age stats


- sales for each genre
- sales vs tags
- platform availability
- 


"""

##
main_df['required_age'].value_counts()

##
df_ = main_df.loc[main_df['required_age'] > 15, ['name', 'owners_est']] \
    .sort_values('owners_est', ascending=False)
df_.head(20)

##
print(f"Estimated ??? {df_['owners_est'].sum()/1e6:.0f}M")
print(f"Estimated share of 16+ games: {df_['owners_est'].sum() / main_df['owners_est'].sum():.4f}")

"""
Although most games have no age requirements, some superproductions have an age requirement.
In terms of market share, this represents an estimated 4.3% of market share for those games.
"""

##
languages_df.sum().sort_values(ascending=False).head(20)

##
df_ = main_df.loc[languages_df.sum(axis=1)>10, ['name', 'owners_est']]
df_.sort_values('owners_est', ascending=False).head(20)

##
df_['owners_est'].sum()

"""
Almost all games are available in english. There is a gap 
About 20% of them are also available in other european languages (french, german, italian, etc).


Translation to subtitles, not necessarily of voice
"""



# %% Genres

"""
## <a name="genres"></a> Genres analysis

OK - most represented in terms of nb games
OK - best rated genres
OK - genres by publishers
OK - number of players by genres
# - most lucrative genres (nb players * price)
OK - genres evolution
"""

##
genres_df.sum().sort_values(ascending=False)

"""
Mostly independent games
action, adventure, strategy, simulation, RPG, sports, racing
"""

##
grades = main_df['positive'] / (main_df['positive'] + main_df['negative'])
weights = (main_df['positive'] + main_df['negative']).apply(np.sqrt)

genre_grades = {}
for genre_name, genre in genres_df.items():
    w = weights.loc[genre]
    genre_grades[genre_name] = (grades.loc[genre] * w).sum() / w.sum()
grades_df = pd.Series(genre_grades, name='grade')
grades_df.sort_values(ascending=False)

"""
Generally positive
"""

##
def genres_by_publisher(publisher: str)-> pd.DataFrame:
    df_ = genres_df[main_df['publisher'] == publisher]
    return df_.sum()

genres_by_publisher('Valve')
genres_by_publisher('Ubisoft')


##
genre_owners = {}
for genre_name, genre in genres_df.items():
    genre_owners[genre_name] = main_df.loc[genre, 'owners_est'].sum()
genre_owners_df = pd.Series(genre_owners, name='grade')
genre_owners_df.sort_values(ascending=False)

"""
Action and advrenture are the most popular genre. Independent (`'Indie'`) games take
a significant market share.
"""

## 
genre_release_df = genres_df.set_index(release_dates)
genre_releases_per_month = genre_release_df.resample('MS').sum()
cumulative_genre_releases = genre_releases_per_month.cumsum()

##
COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]

fig6, axs6 = plt.subplots(
    nrows=1, ncols=2, figsize=(8, 5), dpi=100,
    gridspec_kw={'left': 0.08, 'right': 0.97, 'top': 0.57, 'bottom': 0.1, 'wspace': 0.25})
fig6.suptitle('Figure 6: Evolution of game genre releases', x=0.02, ha='left')

polys = axs6[0].stackplot(genre_releases_per_month.index, genre_releases_per_month.T,
                  colors=COLORS)
axs6[0].set_xlim(12410, 19730)
axs6[0].xaxis.set_major_locator(mdates.YearLocator(4))
axs6[0].set_ylim(0, 2500)
axs6[0].set_yticks(np.linspace(0, 2500, 6), np.linspace(0, 2.5, 6))
axs6[0].grid(visible=True)
axs6[0].set_xlabel('Date')
axs6[0].set_ylabel('Monthly game releases (x 1000)')



axs6[1].stackplot(cumulative_genre_releases.index, cumulative_genre_releases.T,
                  baseline='zero', colors=COLORS)
axs6[1].set_xlim(12410, 19730)
axs6[1].xaxis.set_major_locator(mdates.YearLocator(4))
axs6[1].set_ylim(0, 160000)
axs6[1].set_yticks(np.linspace(0, 160000, 9), np.arange(0, 180, 20))
axs6[1].grid(visible=True)
axs6[1].set_xlabel('Date')
axs6[1].set_ylabel('Cumulative game releases (x 1000)')

fig6.legend(handles=polys, labels=genre_release_df.columns.to_list(),
            ncols=4, loc=(0.02, 0.59))

plt.show()

"""
The increasing trend in games release is the same for all the major genres.
"""


# %% Platform

"""
## <a name="platform"></a> Platform analysis


OK - general platform availability
OK - nb owners and platforms
OK - genres and platforms
# - price and platform
- platform availability evolution tendency (window release date and look at platform availability)
"""

##
platforms_df.sum()

##
platforms_df.sum() / len(platforms_df)


"""
Almost all games are available on Windows, while only 23% and !%% of the games are available on Mac and linux, respectively.
"""

##
# main_df.loc[platforms_df['linux'], ['name', 'owners_est']].sum() # .sort_values('owners_est', ascending=False).head(20)
df_ = main_df.loc[:, ['owners_est']*3].set_axis(platforms_df.columns, axis=1)
df_ = df_.mask(~platforms_df.to_numpy(), other=0.).sum()
df_

##
df_ / main_df['owners_est'].sum()

"""
Although only 15-20% of the games are available to Mac and Linux users, the availability rises to 30-40%
in terms of games usage basis. This means that popular games tend to be available on Mac and Linux.
"""

##
genre_availability = {}
for platform, mask in platforms_df.items():
    genre_availability[platform] = genres_df.loc[mask].sum()
genre_availability_df = pd.DataFrame(genre_availability)
idx = genre_availability_df.sum(axis=1).sort_values(ascending=False).index
genre_availability_df.loc[idx]

##
(genre_availability_df / genres_df.sum().to_numpy()[:, None]).loc[idx]

"""
Independent and strategy games have a larger availability on Mac and Linux platforms than average.
"""

##
platform_release_df = platforms_df.set_index(release_dates)
platform_releases_per_month = platform_release_df.resample('MS').sum()
cumulative_platform_releases = platform_releases_per_month.cumsum()

##
fig7, axs7 = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 3.8), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.97, 'top': 0.83, 'bottom': 0.13, 'wspace': 0.25})
fig7.suptitle('Figure 7: Evolution of game releases in different platforms', x=0.02, ha='left')

polys = axs7[0].stackplot(platform_releases_per_month.index, platform_releases_per_month.T)
axs7[0].set_xlim(12410, 19730)
axs7[0].xaxis.set_major_locator(mdates.YearLocator(4))
axs7[0].set_ylim(0, 1200)
axs7[0].set_yticks(np.linspace(0, 1200, 7))
axs7[0].grid(visible=True)
axs7[0].set_xlabel('Date')
axs7[0].set_ylabel('Monthly game releases (x 1000)')



axs7[1].stackplot(cumulative_platform_releases.index, cumulative_platform_releases.T)
axs7[1].set_xlim(12410, 19730)
axs7[1].xaxis.set_major_locator(mdates.YearLocator(4))
axs7[1].set_ylim(0, 80000)
axs7[1].set_yticks(np.linspace(0, 80000, 9), np.arange(0, 90, 10))
axs7[1].grid(visible=True)
axs7[1].set_xlabel('Date')
axs7[1].set_ylabel('Cumulative game releases (x 1000)')

fig7.legend(handles=polys, labels=platform_release_df.columns.to_list(),
            ncols=3, loc=(0.3, 0.86))

plt.show()

"""
The trend is stable over time.
"""



