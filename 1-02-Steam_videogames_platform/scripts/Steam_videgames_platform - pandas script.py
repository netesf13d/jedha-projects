# -*- coding: utf-8 -*-
"""
Script version of Steam videogames platform project, using pandas for data
treatment.
"""

import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



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


"""
### Column parsing and preprocessing
"""

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

r"""
The owner ranges are specified on a logarithmic scale (each successive range increases in size by a factor $\sim 2$).
In order to preserve such scaling, we estimate the number of owners from the two bounds by taking their *geometric* mean.
This is problematic for the lowest range (0 - 20000), which we estimate by $\sqrt{20000} \simeq 141$.
We must keep in mind that this last value probably underestimates the actual number of owners.
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
"""
### Construction of intermediate dataframes
"""

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
def parse_languages(langs: str) -> set[str]:
    """
    Parse language strings into a list of languages.
    """
    lang_list = langs.replace('[b]*[/b]', '') \
                     .replace('\r\n', ',') \
                     .replace(', ', ',') \
                     .split(',')
    return set(l for lang in lang_list if (l:=lang.strip()))

languages = [parse_languages(lang) for lang in raw_df['languages']]
all_languages = set.union(*languages)
languages_df = pd.DataFrame.from_dict(
    {lang: [(lang in langs) for langs in languages] for lang in all_languages})


# %% Macro
"""
## <a name="macro"></a> Macro-level analysis

We begin our study by analyzing the market globally. We will focus on the following:
- Games popularity and revenues
- Publishers and developers popularity and revenues
- Evolution of the market in terms of game releases
- Games public and availability


### Games popularity and revenues

There are interesting patterns in game usage, price and revenues which we highlight here.
"""

## dataframe to study owners and revenues
df_ = main_df.loc[:, ['name', 'owners_low', 'owners_est', 'price']]
df_ = df_.assign(revenues_low=df_['price']*df_['owners_low'],
                 revenues_est=df_['price']*df_['owners_est'])
print(df_.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99, 0.999]))

"""
The basic statistics shown here reveal interesting characteristics of the game market.
- The average game price is below 10$. Actually, 7780 games are free. Such games actually generate a major part of their revenue
in-game microtransactions. This is an important aspect of the video game industry that we will not be able to analyze here.
- Concerning game owners and revenues, the mean and the median differ by 3 orgers of magnitude. The market is thus extremely inhomogeneous.
"""


##
df_.sort_values('owners_est', ascending=False).iloc[:20]

##
df_.sort_values('revenues_est', ascending=False).iloc[:20]

"""
Among the most popular games, we recognize well known games and franchises (Elden Ring, Grand Theft Auto, The Witcher, etc). We also note that among
them, more than half are free to play. This shows that microtransactions are an important part of
games revenues from Steam's platform. We will not be able to analyze this source of revenues with this dataset.
"""

##
fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.95, 'top': 0.89, 'bottom': 0.14})
fig1.suptitle('Figure 1: Game prices distribution', x=0.02, ha='left')

ax1.hist(df_['price'], bins=np.linspace(0, 100, 51))
ax1.set_xlim(0, 80)
ax1.set_ylim(0, 20000)
ax1.set_yticks(np.linspace(0, 20000, 11), np.arange(0, 21, 2))
ax1.grid(visible=True)
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Number of owners (x1000)')

plt.show()

"""
Figure 1 shows the distribution of prices. We recover the fact that game prices are low in general.
We note that prices have preferential values: 4.99, 9.99, 14.99, etc.
"""


## Cumulative game owners distribution function
owners_distrib = df_['owners_low'].value_counts() / df_['owners_low'].count()
owners_cdf = owners_distrib.iloc[::-1].cumsum().iloc[::-1]


## Share of total game usage vs number of games sorted by decreasing number of owners
# here we use our owners estimate
tot_owners = df_['owners_est'].sum()
cum_owners = (df_['owners_est'].sort_values().cumsum() / tot_owners).to_numpy()
own_frac = np.linspace([1], [0], 201, endpoint=True)
own_nb_games = len(cum_owners) - np.argmax(cum_owners >= own_frac, axis=-1)
own_nb_games[0] = 0

## Cumulative revenues distribution function
revenues_vals = np.logspace(1, 9, 17)
revenues_cdf = [(df_['revenues_low'] >= x).mean() for x in revenues_vals]

## Share of total game revenues vs number of games sorted by decreasing revenue
# here we use our owners estimate to estimate the revenues
tot_revenues = df_['revenues_est'].sum()
cum_revenues = (df_['revenues_est'].sort_values().cumsum() / tot_revenues).to_numpy()
cum_revenues = cum_revenues[cum_revenues > 0]
rev_frac = np.linspace([1], [0], 201, endpoint=True)
rev_nb_games = len(cum_revenues) - np.argmax(cum_revenues >= rev_frac, axis=-1)
rev_nb_games[0] = 0


##
fig2, axs2 = plt.subplots(
    nrows=2, ncols=2, figsize=(7.2, 6), dpi=100,
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.9, 'bottom': 0.1,
                 'wspace': 0.26, 'hspace': 0.42})
fig2.suptitle('Figure 2: Distribution of game owners and revenues', x=0.02, ha='left')
fig2.text(0.5, 0.92, 'Game owners', ha='center', fontsize=11)
fig2.text(0.5, 0.46, 'Game revenues', ha='center', fontsize=11)


axs2[0, 0].plot(owners_cdf.index, owners_cdf.to_numpy(),
                marker='o', linestyle='')
axs2[0, 0].set_xscale('log')
axs2[0, 0].set_xlim(1e2, 1e10)
axs2[0, 0].set_yscale('log')
axs2[0, 0].set_ylim(1e-5, 1)
axs2[0, 0].grid(visible=True)
axs2[0, 0].set_xlabel('Number of owners')
axs2[0, 0].set_ylabel('Fraction of games')
axs2[0, 0].set_title('Compl. cumulative distrib.', fontsize=11)


axs2[0, 1].plot(own_nb_games, 1-own_frac, marker='', linestyle='-')
axs2[0, 1].plot([0, own_nb_games[100], own_nb_games[100]], [0.5, 0.5, 0],
                color='k', lw=0.8, ls='--')
axs2[0, 1].set_xscale('log')
axs2[0, 1].set_xlim(8e-1, 1e5)
axs2[0, 1].set_ylim(0, 1)
axs2[0, 1].grid(visible=True)
axs2[0, 1].set_xlabel('Number of games')
axs2[0, 1].set_ylabel('Fraction of total game usage')


axs2[1, 0].plot(revenues_vals, revenues_cdf, marker='o', linestyle='')
axs2[1, 0].set_xscale('log')
axs2[1, 0].set_xlim(1e1, 1e10)
axs2[1, 0].set_yscale('log')
axs2[1, 0].set_ylim(1e-5, 1)
axs2[1, 0].grid(visible=True)
axs2[1, 0].set_xlabel('Game revenue ($)')
axs2[1, 0].set_ylabel('Fraction of games')
axs2[1, 0].set_title('Compl. cumulative distrib.', fontsize=11)


axs2[1, 1].plot(rev_nb_games, 1-rev_frac, marker='', linestyle='-')
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
Figure 2 shows the some characteristics of the distributions of game owners (top) and game revenues (bottom).
The left panels show the complementary cumulative distribution functions. A data point represent the fraction of games having larger number of owners/revenues
than its abscissa. The first data point at ordinate 1 is not visible.
It corresponds to zero abscissa (more than zero owner/zero revenue), and thus does not appear on the log-log plot.
The right panels show the fraction of total game units downloaded or game revenues as a function of the number of games, sorted by decreasing importance.
These results reveal some important characteristics of the game market:
- The distribution of game owners (and therefore of revenues) is fat tailed.
Above a certain threshold, the number of games with a $n$ owners is roughly proportional to $p(n) = \frac{A}{n^{1+\alpha}}$,
where the exponent $\alpha$ is an empirical parameter of the distribution. The most important consequence is that a very small
fraction of the games concentrates most of the game usage.
- The distribution of revenues shows that only about 25% of the games seem to produce a revenue. This is of course a strong underestimation:
we are missing the revenues from microtransactions, advertisement, etc.
- About 400 games (less than 1 %!) account for half the game downloads. This tendency seems even stronger for the revenues with half of the total revenues
originating from 200 games (although here we are missing parts of the revenues).

This analysis indicates that the videogames market is actually dominated by a few superproductions.
"""

# %%
"""
### Games publishers and developers

We now study how the market is shared among game publishers and developers. We first analyze things in terms
of games released. One important metric in the game industry is the number of units sold, which we analyze in second.


#### Number of game releases
"""

##
publishers = main_df['publisher'].value_counts()
publishers

##
developers = main_df['developer'].value_counts()
developers

"""
There are about 30000 publishers/developers. We recognize some well known companies among top publishers
(SEGA, Square Enix). The developer names are less familiar, but we note the presence of individuals
(Laush Dmitriy Sergeevich, Artur Smiarowski). These are independent developers, that actually take a large
part of the released games.
"""

## Cumulative distributions of number of game releases for developers and publishers
vals = np.logspace(0, 3, 13)
publishers_cdf = [(publishers >= x).sum() for x in vals]
developers_cdf = [(developers >= x).sum() for x in vals]

##
fig3, ax3 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.88, 'bottom': 0.13})
fig3.suptitle('Figure 3: Complementary cumulative distribution of released'
              '\n               games per publisher/developer',
              x=0.02, ha='left')

ax3.plot(vals, publishers_cdf,
         marker='o', linestyle='', alpha=0.8, label='publishers')
ax3.plot(vals, developers_cdf,
         marker='o', linestyle='', alpha=0.7, label='developers')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(visible=True)
ax3.set_xlabel('number of games published')
ax3.set_ylabel('number of publishers/developers')
ax3.legend()

plt.show()

"""
Figure 3 shows the complementary cumulative distribution of games release by publisher/developer.
The interpretation is the same as in figure 2: we represent the number of publshers having more than $x$ games published.
Similarly to game usage, we note that the distribution is fat-tailed. Out of the 30000, more than 20000 developers/publishers released only one game.
Only a dozen released more than 100 games.
"""

"""
#### Games units distributed

We consider here the total number of games sold by (or downloaded from) a publisher/developer.
"""

##
publishers_vs_owners = main_df.loc[:, ['publisher', 'owners_low']] \
                              .groupby('publisher') \
                              .sum() \
                              .sort_values('owners_low', ascending=False)
publishers_vs_owners

##
developers_vs_owners = main_df.loc[:, ['developer', 'owners_low']] \
                              .groupby('developer') \
                              .sum() \
                              .sort_values('owners_low', ascending=False)
developers_vs_owners

"""
We sort the publishers/developers by number of owners of their games.
The top names are all well known companies. The inhomogeneity in terms of owner base is large,
ranging over 7 orders of magnitude. This is a combined effect: the largest companies also attract the largest public.
"""


## Cumulative distributions of number of games distributed for developers and publishers
vals = np.logspace(3, 9, 19)
pub_vs_owners_cdf = [(publishers_vs_owners >= x).sum() for x in vals]
dev_vs_owners_cdf = [(developers_vs_owners >= x).sum() for x in vals]


fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 4), dpi=100,
    gridspec_kw={'left': 0.13, 'right': 0.97, 'top': 0.88, 'bottom': 0.13})
fig4.suptitle('Figure 4: Complementary cumulative distribution of game units'
              '\n               distributed per publisher/developer',
              x=0.02, ha='left')

ax4.plot(vals, pub_vs_owners_cdf,
         marker='o', linestyle='', alpha=0.8, label='publishers')
ax4.plot(vals, dev_vs_owners_cdf,
         marker='o', linestyle='', alpha=0.7, label='developpers')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(visible=True)
ax4.set_xlabel('Total units owned')
ax4.set_ylabel('number of publishers / developers')
ax4.legend()

plt.show()

"""
Figure 4 shows the complementary cumulative distribution of owned games for publishers/developers. We note that out of the 30000 publishers/developers, only about 10000 have actually distributed more than 1000 games units in total.
As expected, we recover a fat-tailed distribution, with a number of units owned that spreads over 7 orders of magnitude. Some companies having "sold" close to 1 billion games in total.
"""


# %%
"""
### Evolution of game releases

A company can take more risks in terms of game production in the context of a healthy game market.
The situation of the latter is therefore an important for our analysis.
In this section, we study it through the evolution of game releases.

A better indication of the market health would be the evolution of the number of players,
which is not available in this dataset.
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
Figure 5 presents the monthly game releases (left panel) and cumulated releases (right panel).
Monthly releases stayed under 50 until 2013, when they increased steadily to 400 in 2017.
This is possibly the result of an increase in popularity of Steam's platform, or a change of some sort in policy.
For instance it could be the possibility for independent game developers to publish their games on the platform.
We then note a jump of about 200 releases in 2017, which stabilizes at 600 during about 3 years. Its cause is unclear.
This is followed, in early 2020 by another jump in game releases. This last increase occurs during the global COVID lockdown.
The number of releases is stable between 700 and 800 since then.
"""



# %%
"""
### General game availability

We conclude this section by analyzing games availability on the platform. This should help getting insights
about the target audience for a new game. We focus here on age restrictions and language availabilty.
"""

##
main_df['required_age'].value_counts()

##
df_ = main_df.loc[main_df['required_age'] > 15, ['name', 'owners_est']] \
    .sort_values('owners_est', ascending=False)
df_.head(20)

##
print("Estimated number of age-restricted games (16+ yo) sold:",
      f"{df_['owners_est'].sum()/1e6:.0f} M")
print("Estimated market share of age-restricted games:",
      f"{100 * df_['owners_est'].sum() / main_df['owners_est'].sum():.4} %")

"""
Although most games have no age requirements, some superproductions do have an age limit.
Age-restricted games represent an estimated 4.3% of the global market share.
Producing an age-restricted game has both advantages and disadvantages. On the one hand, the potential customer basis
is mechanically reduced. On the other hand, it allows producers to introduce more features and mechanics in their games.
"""

##
languages_df.sum().sort_values(ascending=False).head(20)

##
df_ = main_df.loc[languages_df.sum(axis=1)>10, ['name', 'owners_est']]
df_.sort_values('owners_est', ascending=False).head(20)

##
print("Estimated number of games sold available in 10+ languages:",
      f"{df_['owners_est'].sum()/1e6:.0f} M")
print("Estimated market share of games available in 10+ languages:",
      f"{100 * df_['owners_est'].sum() / main_df['owners_est'].sum():.4} %")


"""
Almost all games are available in english. There is a large gap in language availability between english and other
languages. Only about 20% of them are also available in other european languages (french, german, italian, etc).
However, those games that attract many players also are available in many languages. About 40% of games downloaded
are available in more than 10 languages.

It is important to note that language availability does not necessarily means translation of in-game voices,
the availability may be limited to menu translation and subtitles.
"""



# %% Genres
"""
## <a name="genres"></a> Genres analysis

Some game genres are more popular than others, or more expensive to develop. Some developers/publishers have more expertise
in some specific genres. The choice of the genre for a game release must take these factors into account.
We now analyze the market from the games genres perspective.
"""

##
genres_df.sum().sort_values(ascending=False)

"""
Most games are from independent developers/publishers (`'Indie'` genre). This is expected since those games are
in general faster to produce, and their development is acessible to many people.
The most proeminent genres are the action-adventure games. Casual games, aiming at a broad
public rather than the hobbyist player, are also proeminent. This is explained by the fact
those, like independent games, are smaller and thus faster to produce.
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
We compute the genres grades by averaging the number of positive reviews, weighted with the number of reviews for each game. All genres receive generally positive reviews.
"""


##
def genres_by_developers(developers: list[str])-> pd.DataFrame:
    """
    Numbers of games released in each genre category for the given publishers.
    """
    df_ = genres_df.assign(developer=main_df['developer'])
    df_ = df_[df_['developer'].apply(lambda x: x in developers)]
    return df_.groupby('developer').sum().T

def genres_by_publishers(publishers: list[str])-> pd.DataFrame:
    """
    Numbers of games released in each genre category for the given developers.
    """
    df_ = genres_df.assign(publisher=main_df['publisher'])
    df_ = df_[df_['publisher'].apply(lambda x: x in publishers)]
    return df_.groupby('publisher').sum().T

##
genres_by_publishers(['Valve', 'Ubisoft'])

##
genres_by_developers(['Bethesda Game Studios', 'CAPCOM Co., Ltd.'])

"""
We can see that some publishers are specialized in some genres. For instance, Valve published
almost exclusively action games. Ubisoft, on the other hand, is more diversified. Developers
also specialize in some game genres. Bethesda produces mainly RPGs, while Capcom delelops action-adventure games.
"""

##
genre_owners = {}
for genre_name, genre in genres_df.items():
    genre_owners[genre_name] = main_df.loc[genre, 'owners_est'].sum()
genre_owners_df = pd.Series(genre_owners, name='grade')
genre_owners_df.sort_values(ascending=False)

"""
Action and adventure are the most popular genres. Independent games also take
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
axs6[0].set_ylabel('Monthly releases (x 1000)')


axs6[1].stackplot(cumulative_genre_releases.index, cumulative_genre_releases.T,
                  baseline='zero', colors=COLORS)
axs6[1].set_xlim(12410, 19730)
axs6[1].xaxis.set_major_locator(mdates.YearLocator(4))
axs6[1].set_ylim(0, 160000)
axs6[1].set_yticks(np.linspace(0, 160000, 9), np.arange(0, 180, 20))
axs6[1].grid(visible=True)
axs6[1].set_xlabel('Date')
axs6[1].set_ylabel('Cumulative releases (x 1000)')

fig6.legend(handles=polys, labels=genre_release_df.columns.to_list(),
            ncols=4, loc=(0.02, 0.59))

plt.show()

"""
Figure 7 shows stack plots of monthly genre releases (left panel) and cumulated releases (right panel) over time.
The counts are larger than the number of games since the latter can have multiple genres. The incresaing trend
in game releases appear to be balanced between all the major genres. No genre is gaining interest over time.
"""


# %% Platform
"""
## <a name="platform"></a> Platform analysis

The choice of platform availability is important in game production. Being available in different platforms brings more public,
but this may not be worth the extra costs (eg licences and software adaptation). We therefore conclude our study of the game market
by focusing on the importance of the different platform available: Windows, Mac and Linux.
"""

##
platforms_df.sum()

##
platforms_df.mean()


"""
Almost all games are available on Windows, while only 23% and 15% of the games are available on Mac and linux, respectively.
This makes Windows a mandatory platform for computer games.
"""

##
df_ = main_df.loc[:, ['owners_est']*3].set_axis(platforms_df.columns, axis=1)
df_ = df_.mask(~platforms_df.to_numpy(), other=0.).sum()
df_

##
df_ / main_df['owners_est'].sum()

"""
Although only 15/23% of the games are available to Linux and Mac users, the availability rises to 33/41%
in terms of downloaded games. This means that popular games tend to be available on Mac and Linux.
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
Figure 7 shows stack plots of monthly game releases over time (left panel) and their cumulated number (right panel).
The trend is stable over time. The different pLatforms availability does not change significantly over time.
No platform seems to be taking over the others.
"""

# %% Conclusion
"""
## <a name="conclusion"></a> Conclusion and perspectives

We can summarize the results of our study of Steam's game marketplace with the following points:
- In terms of shares, the market is dominated by a few superproductions from major publisher companies.
These games are played by many players and generate the largest revenues. This market structure is actually
quite general and occurs accross the whole entertainment inductry (music, movies, series, etc).
- About half publishers/developers released only one game. These are independent people or very small companies.
Similarly to the revenues, the game release-by-company distribution is fat tailed,
with only about 10 companies having released more than 100 games.
- This structure translates to the distribution of the company's customer base. The amplitude of variation here is even larger,
considering that games released by large comanies also have more players.
- The game releases seem stable since early 2020 with about 700-800 monthly releases.
- The genre and platform availability of released games is stable over time


In terms of strategy for a new videogame from an important company such as Ubisoft, we can provide the following advices:
- The company has enough ressources to make a superproduction, which is definitely an option to consider.
However, the associated production costs are also large and so are the financial risks.
- For a video game with a completely new gameplay or concept, the risks might be unacceptable. For
such a game, one could consider being less ambitious and test the concept on a smaller scale, for instance by
publishing a game with limited production costs.
- In terms of genres, one should focus on the main genres: action adventure, strategy, RPG, etc. and choose from
those in which the company has expertise.
- For a large production, translating the game is a necessity. Adapting the game to many platforms is also
strategic since the associated costs are low as compared to the other production costs.
The gains in players likely outweights the adaptation costs.


Our analysis has some limitations, some of which we mentioned in the introduction. We recall the most important ones:
- We have very limited information on the actual game usage. The number of game owners is provided as a rather braod range
which forced us to construct a very rough estimate. We do not now whether people actually play to the game, nor how
many time they spend playing.
- The number of game owners corresponds to the values at the time when the datset was built. We have no information
about its evolution in time, which would be useful to make prospects.
- Our data are limited to computer platforms. it would be very beneficial to a market analysis to include video games usage in phones or consoles.
- Even in the scope of computer platforms, we are only considering Steam's marketplace.
We miss information for video games not directly available on Steam (for instance, Fortnite, one on the most played games).
"""

