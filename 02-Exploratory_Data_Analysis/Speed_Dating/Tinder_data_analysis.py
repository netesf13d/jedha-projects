#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
from io import BytesIO


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def to_float(series: pd.Series)-> np.ndarray:
#     """
#     Convert a series of floats loaded as objects into an array of floats.
#     The floats in the raw dataset have format 'xxx,xxx.xx'The floats in the raw dataset have format 'xxx,xxx.xx', the comma prevents
#     automatic casting to float.
#     """
#     arr = [x if isinstance(x, float) else float(x.replace(',', ''))
#            for x in series]
#     return np.array(arr)



# =============================================================================
# 
# =============================================================================

fname = "./Speed+Dating+Data.csv"
df = pd.read_csv(fname, encoding='mac_roman', thousands=',', decimal='.')



subject_entries = pd.concat([pd.Series([True], index=['iid']),
                             df.groupby('iid').nunique().max() <= 1])
partner_entries = ~subject_entries


subject_cols = df.columns[subject_entries]
subject_df = df[subject_cols].groupby('iid').aggregate(lambda x: x.iloc[0])
# subject_df.loc[:, 'race'] = subject_df['race'].astype(int)


# %%

cols = ['age', 'income', 'mn_sat']
for c in cols:
    if sum(~np.isnan(subject_df[c])) > 276:
        print(c)


# %% age and income histograms

def textstr(df: pd.DataFrame)-> str:
    text = (f'Mean   : {df.mean():.2f}\n'
            f'Std dev : {df.std():.2f}'
            f'\nMedian : {df.median():.2f}')
    return text


age = subject_df['age']
income = subject_df['income']
mn_sat = subject_df['mn_sat']


fig1, axs1 = plt.subplots(
    nrows=1, ncols=3, figsize=(9, 4), dpi=96,
    gridspec_kw={'left': 0.06, 'right': 0.97, 'top': 0.85, 'bottom': 0.13, 'wspace': 0.18})
fig1.suptitle('Figure 1: ', x=0.02, ha='left')

for ax in axs1:
    ax.grid(visible=True, linewidth=0.3)
    ax.grid(visible=True, which='minor', linewidth=0.3)
    

axs1[0].set_title("Age distribution")
axs1[0].hist(subject_df['age'], bins=np.linspace(10, 60, 51))
axs1[0].set_xlim(10, 60)
axs1[0].set_xlabel("Age (years)")
axs1[0].set_ylabel('Counts')
axs1[0].text(0.58, 0.825, textstr(subject_df['age']),
             transform=axs1[0].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs1[1].set_title("Income distribution")
axs1[1].hist(subject_df['income'], bins=np.linspace(1e4, 1.2e5, 23))
axs1[1].set_xlim(0, 1.2e5)
axs1[1].set_xticks(np.linspace(0, 120000, 7), np.arange(0, 130, 20))
axs1[1].set_xlabel("Median household income (k$/year)")
axs1[1].text(0.58, 0.825, textstr(subject_df['income']/1000),
             transform=axs1[1].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs1[2].set_title("Median SAT score distribution")
axs1[2].hist(subject_df['mn_sat'], bins=np.linspace(900, 1500, 25))
axs1[2].set_xlim(800, 1600)
axs1[2].set_xticks([900, 1100, 1300, 1500], minor=True)
axs1[2].set_xlabel("Median SAT score")
axs1[2].text(0.035, 0.825, textstr(subject_df['mn_sat']),
             transform=axs1[2].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

plt.show()

"""
Subjects are students at Columbia university hence realtively young
The median household income is rather low, as expected for students which have generally not started their professional life.
The SAT scores are higher than average, as expected for people who follow graduate studies.
"""



# %% Racial stats

##
racial_df = subject_df.loc[:, ['race', 'gender', 'age', 'income']]

races = {1.0: 'Black', 2.0: 'Caucasian', 3.0: 'Latino',
         4.0: 'Asian', 5.0: 'Nat. American', 6.0: 'Other'}
genders = {0: 'Female', 1: 'Male'}
racial_df = racial_df.astype({'race': object, 'gender': object})
racial_df.loc[:, 'race'] = racial_df['race'].replace(races)
racial_df.loc[:, 'gender'] = racial_df['gender'].replace(genders)
racial_df

##
racial_df.groupby(['race', 'gender']).count()

"""
More caucasians than the rest.
"""

##
ages, incomes = {}, {}
for race, df_ in racial_df.groupby('race'):
    
    vals_f = df_.loc[df_['gender'] == 'Female', 'age']
    vals_m = df_.loc[df_['gender'] == 'Male', 'age']
    ages[race] = [vals_f[~np.isnan(vals_f)], vals_m[~np.isnan(vals_m)]]
    
    vals_f = df_.loc[df_['gender'] == 'Female', 'income']
    vals_m = df_.loc[df_['gender'] == 'Male', 'income']
    incomes[race] = [vals_f[~np.isnan(vals_f)], vals_m[~np.isnan(vals_m)]]


fig2, axs2 = plt.subplots(
    nrows=2, ncols=1, figsize=(5.5, 5.2),
    gridspec_kw={'left': 0.14, 'right': 0.96, 'top': 0.92, 'bottom': 0.06, 'hspace': 0.08})
fig2.suptitle("Figure 2: Box plots",
              x=0.02, ha='left')


for i, (race, race_ages) in enumerate(ages.items()):
    bplot = axs2[0].boxplot(race_ages, positions=[i+0.82, i+1.18],
                            widths=0.24, patch_artist=True,
                            medianprops={'color': 'k'})
    for patch, color in zip(bplot['boxes'], ['#1F77B480', '#FF7F0E80']):
        patch.set_facecolor(color)
axs2[0].set_xlim(0.5, 5.5)
axs2[0].tick_params(bottom=False, labelbottom=False)
axs2[0].set_ylim(15, 60)
axs2[0].set_ylabel('Age (years)', labelpad=10)
axs2[0].grid(visible=True, axis='y', linewidth=0.3)
axs2[0].legend(handles=bplot['boxes'], labels=['Female', 'Male'], ncols=2)


for i, (race, race_incomes) in enumerate(incomes.items()):
    bplot = axs2[1].boxplot(race_incomes, positions=[i+0.82, i+1.18],
                            widths=0.24, patch_artist=True,
                            medianprops={'color': 'k'})
    for patch, color in zip(bplot['boxes'], ['#1F77B480', '#FF7F0E80']):
        patch.set_facecolor(color)

axs2[1].set_xlim(0.5, 5.5)
axs2[1].set_xticks(np.arange(1, 6), list(ages.keys()))
axs2[1].set_ylim(0, 120000)
axs2[1].set_yticks(np.linspace(0, 120000, 7), np.arange(0, 130, 20))
axs2[1].set_ylabel('Median household income\n(k$/year)', labelpad=2)
axs2[1].grid(visible=True, axis='y', linewidth=0.3)
axs2[1].legend(handles=bplot['boxes'], labels=['Female', 'Male'], ncols=2)

plt.show()

"""
Salaries in the same range except maybe black females but fluctuations.
"""


# %% Interests

# TODO aggregate genders and just discuss mean differences
interests = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
             'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
             'movies', 'concerts', 'music', 'shopping', 'yoga']
interests_df = subject_df.loc[:, ['gender'] + interests]
mean_interests = interests_df.groupby('gender').mean()
interests_df.describe()

##
interests_df = interests_df.astype({'gender': object})
interests_df.loc[:, 'gender'] = interests_df['gender'].replace(genders)
interests_data = {}
for gender, df_ in interests_df.groupby('gender'):
    cols = [inter.to_numpy() for _, inter in df_.iloc[:, 1:].items()]
    interests_data[gender] = [col[~np.isnan(col)] for col in cols]
# ages, incomes = {}, {}
# for (race,), df_ in racial_df.groupby(['race']):
    
#     vals_f = df_.loc[df_['gender'] == 'Female', 'age']
#     vals_m = df_.loc[df_['gender'] == 'Male', 'age']
#     ages[race] = [vals_f[~np.isnan(vals_f)], vals_m[~np.isnan(vals_m)]]
    
#     vals_f = df_.loc[df_['gender'] == 'Female', 'income']
#     vals_m = df_.loc[df_['gender'] == 'Male', 'income']
#     incomes[race] = [vals_f[~np.isnan(vals_f)], vals_m[~np.isnan(vals_m)]]


fig3, axs3 = plt.subplots(
    nrows=2, ncols=1, figsize=(7.5, 5.2),
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.92, 'bottom': 0.11, 'hspace': 0.08})
fig3.suptitle("Figure 3: Violin plots",
              x=0.02, ha='left')


vplot = axs3[0].violinplot(
    interests_data['Female'][:17], widths=0.6, showmeans=True)
for i, m in enumerate(mean_interests.T[1]):
    axs3[0].plot([i-0.2, i+0.2], [m, m])

axs3[0].set_xlim(0.5, 17.5)
axs3[0].tick_params(bottom=False, labelbottom=False)
axs3[0].set_ylim(0, 14)
axs3[0].set_ylabel('Score', y=-0.05, labelpad=10)
axs3[0].grid(visible=True, axis='y', linewidth=0.3)
axs3[0].legend(handles=[vplot['bodies'][0]], labels=['Female'], ncols=2)


vplot = axs3[1].violinplot(
    interests_data['Male'], widths=0.6, showmeans=True)
for elt in vplot['bodies']:
    elt.set_facecolor('#FF7F0E')
for elt in ['cbars', 'cmaxes', 'cmins', 'cmeans']:
    vplot[elt].set_edgecolor('#FF7F0E')

axs3[1].set_xlim(0.5, 17.5)
axs3[1].set_xticks(np.arange(1, 18), interests, rotation=30)
axs3[1].set_ylim(0, 14)
# axs3[1].set_yticks(np.arange(0, 16, 2))
# axs3[1].set_ylabel('Median household income\n(k$/year)', labelpad=2)
axs3[1].grid(visible=True, axis='y', linewidth=0.3)
axs3[1].legend(handles=[vplot['bodies'][0]], labels=['Male'], ncols=2)

plt.show()

"""
Choose violin plots because distrib bounded
"""

# %%

"""
Dealing with attributes is more difficult because its a mess.
Before
1_1 : what I look for ; waves 6-9 10pts scale, others 100 pts
2_1 : what the other looks for ; waves 6-9 10pts scale, others 100 pts total
3_1 : self assessment ; 10pts scale
4_1 : what my fellows look for; waves 6-9 10pts scale, others 100 pts total
5_1 : peceived by others ; 10pts scale
During
1_s : what I look for ; 10 pts scale
3_s : self assessment ; 10pts scale
After
1_2 : what I look for ; waves 6-9 10pts scale, others 100 pts
2_2 : what the other look for ; 100 pts total
3_2 : self assessment ; 10pts scale
4_2 : what my fellows look for; 100 pts total
5_2 : peceived by others ; 10pts scale
7_2 : attributes importance ; 100 pts total
Follow up
1_3 : what I look for ; 100 pts total
2_3 : what the other look for ; 10 pt scale
3_3 : self assessment ; 10pts scale
4_3 : what my fellows look for; 10pts scale
5_3 : peceived by others ; 10pts scale
7_3 : attributes importance ; 100 pts total
"""

# these represent intrinsic attributes, ie exclude interest sharing
attributes = ['Attractive', 'Sincere', 'Intelligent', 'Fun', 'Ambitious']
attr_kw = ['attr', 'sinc', 'intel', 'fun', 'amb']

##
cols = [[kw + f'{i}_{j}' for i in range(1, 6) for kw in attr_kw]
        for j in range(1, 4)]
data = {'Before dating': subject_df.loc[:, cols[0]].count().to_numpy(),
        'After dating': subject_df.loc[:, cols[1]].count().to_numpy(),
        'Follow-up': subject_df.loc[:, cols[2]].count().to_numpy()}
df_ = pd.DataFrame.from_dict(data)
df_.index = [kw + f'{i}' for i in range(1, 6) for kw in attr_kw]
df_

"""
Less responses over time
"""

#%%
"""
Focus on the self assessment: same scale, 4 evals.
"""
##
cols = [kw + f'3_{j}' for j in [1, 's', 2, 3] for kw in attr_kw]
self_eval_df = subject_df.loc[:, cols]
self_eval_df.count()

# 4 sets of self assessed attributes
self_attrs = np.array(np.split(self_eval_df.to_numpy(), 4, axis=1))
diffs = [self_attrs[i+1] - self_attrs[i] for i in range(3)]
diffs = [d[~np.any(np.isnan(d), axis=1)] for d in diffs]

diffs = [self_attrs[2] - self_attrs[0]]
diffs = [d[~np.any(np.isnan(d), axis=1)] for d in diffs]
# diffs = [(df_.iloc[:, 5*(i+1):5*(i+2)] - df_.iloc[:, 5*i:5*(i+1)]).to_numpy()
#          for i in range(3)]

# fig: check that acceptable

fig4, axs4 = plt.subplots(
    nrows=1, ncols=5, figsize=(7.5, 4.2),
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.92, 'bottom': 0.11, 'hspace': 0.08})
fig4.suptitle("Figure 3: Violin plots",
              x=0.02, ha='left')

for i, ax in enumerate(axs4):
    vplot = ax.boxplot(
        [d[:, i] for d in diffs], widths=0.6, bootstrap=10)
# for i, m in enumerate(mean_interests.T[1]):
#     axs4[0].plot([i-0.2, i+0.2], [m, m])

# axs4[0].set_xlim(0.5, 17.5)
# axs4[0].tick_params(bottom=False, labelbottom=False)
# axs4[0].set_ylim(0, 14)
# axs4[0].set_ylabel('Score', y=-0.05, labelpad=10)
# axs4[0].grid(visible=True, axis='y', linewidth=0.3)
# axs4[0].legend(handles=[vplot['bod~es'][0]], labels=['Female'], ncols=2)

plt.show()


#%%
"""
Define scope
- Rescale => 100 pts total
- Average what can be
- Focus on gender differences in terms of attribues

- Study the differences between what I think the other looks for and what he/she actually looks for
-> only in terms of average


- Differences of look for between M/F
  -> how I perceive myself
  -> what I look for (!!! add 'shar' attr)
"""

# self-evauated attributes
se_iids = self_eval_df.index.to_numpy()[~np.all(np.isnan(self_attrs), axis=(0, 2))]
se_attrs = self_attrs[:, ~np.all(np.isnan(self_attrs), axis=(0, 2))]
se_attrs = np.mean(se_attrs, axis=0, where=~np.isnan(se_attrs))

# partner-evaluated attributes
pe_iids, pe_attrs = [], []
partner_eval_df = df.loc[:, ['iid'] + attr_kw]
for iid, df_ in partner_eval_df.groupby('iid'):
    attr_eval = df_.loc[:, attr_kw].to_numpy()
    mask = ~np.isnan(attr_eval)
    if np.all(np.any(mask, axis=0)):
        pe_iids.append(iid)
        pe_attrs.append(np.mean(attr_eval, axis=0, where=mask))
pe_iids, pe_attrs = np.array(pe_iids), np.array(pe_attrs)

# evaluation differences
se_idx = np.isin(se_iids, pe_iids, assume_unique=True)
pe_idx = np.isin(pe_iids, se_iids, assume_unique=True)
eval_diff = se_attrs[se_idx] - pe_attrs[pe_idx]


## plot
fig4, axs4 = plt.subplots(
    nrows=1, ncols=5, figsize=(7.5, 4.2),
    gridspec_kw={'left': 0.1, 'right': 0.96, 'top': 0.92, 'bottom': 0.11, 'hspace': 0.08})
fig4.suptitle("Figure 3: Violin plots",
              x=0.02, ha='left')

for i, ax in enumerate(axs4):
    vplot = ax.boxplot(
        [d[:, i] for d in diffs], widths=0.6, bootstrap=10)



#%%
"""
Construct pair evaluation: attr eval by dating partners
-> diff with self assessment
-> diff with 
"""







# %%

pair_cols = [c for c in df.columns if c not in subject_df.columns]





# %%
"""
- Age diff in matches distrib
- Race match matrix
- nb likes vs each attr
- nb likes vs income
- primality effect: nb likes vs 'order'
- matches vs 'int_corr'
"""










