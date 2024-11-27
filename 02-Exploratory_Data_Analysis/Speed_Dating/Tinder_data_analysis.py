#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
from io import BytesIO


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# def to_float(series: pd.Series)-> np.ndarray:
#     """
#     Convert a series of floats loaded as objects into an array of floats.
#     The floats in the raw dataset have format 'xxx,xxx.xx'The floats in the raw
# dataset have format 'xxx,xxx.xx', the comma prevents
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

pair_cols = [c for c in df.columns if c not in subject_df.columns]


# %% age and income histograms

def textstr(data: np.ndarray)-> str:
    """
    Make a short summary text (mean, std dev, median) to display.
    """
    data = data[~np.isnan(data)]
    text = (f'Mean   : {np.mean(data):.2f}\n'
            f'Std dev : {np.std(data):.2f}'
            f'\nMedian : {np.median(data):.2f}')
    # text = (f'Mean   : {df.mean():.2f}\n'
    #         f'Std dev : {df.std():.2f}'
    #         f'\nMedian : {df.median():.2f}')
    return text


age = subject_df['age']
income = subject_df['income']
mn_sat = subject_df['mn_sat']


fig1, axs1 = plt.subplots(
    nrows=1, ncols=3, figsize=(9, 4), dpi=200,
    gridspec_kw={'left': 0.06, 'right': 0.97, 'top': 0.85, 'bottom': 0.13, 'wspace': 0.18})
fig1.suptitle('Figure 1: Age, income and SAT scores distribution', x=0.02, ha='left')

for ax in axs1:
    ax.grid(visible=True, linewidth=0.3)
    ax.grid(visible=True, which='minor', linewidth=0.3)
    

axs1[0].set_title("Age distribution")
axs1[0].hist(subject_df['age'], bins=np.linspace(10, 60, 51))
axs1[0].set_xlim(10, 60)
axs1[0].set_xlabel("Age (years)")
axs1[0].set_ylabel('Counts')
axs1[0].text(0.575, 0.84, textstr(subject_df['age'].to_numpy()),
             transform=axs1[0].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs1[1].set_title("Income distribution")
axs1[1].hist(subject_df['income'], bins=np.linspace(1e4, 1.2e5, 23))
axs1[1].set_xlim(0, 1.2e5)
axs1[1].set_xticks(np.linspace(0, 120000, 7), np.arange(0, 130, 20))
axs1[1].set_xlabel("Median household income (k$/year)")
axs1[1].text(0.575, 0.84, textstr(subject_df['income'].to_numpy()/1000),
             transform=axs1[1].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs1[2].set_title("Median SAT score distribution")
axs1[2].hist(subject_df['mn_sat'], bins=np.linspace(900, 1500, 25))
axs1[2].set_xlim(800, 1600)
axs1[2].set_xticks([900, 1100, 1300, 1500], minor=True)
axs1[2].set_xlabel("Median SAT score")
axs1[2].text(0.035, 0.84, textstr(subject_df['mn_sat'].to_numpy()),
             transform=axs1[2].transAxes, fontsize=9,
             bbox={'boxstyle': 'round', 'facecolor': '0.92'})

plt.show()

"""
Subjects are students at Columbia university hence realtively young
The median household income is rather low, as expected for students which have generally
not started their professional life.
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
    nrows=2, ncols=1, figsize=(5.5, 5.2), dpi=200,
    gridspec_kw={'left': 0.14, 'right': 0.96, 'top': 0.92, 'bottom': 0.06, 'hspace': 0.08})
fig2.suptitle("Figure 2: Age and income for the different races and genders",
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
"""

"""

interests = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
             'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
             'movies', 'concerts', 'music', 'shopping', 'yoga']
interests_df = subject_df.loc[:, ['gender'] + interests]
mean_interests = interests_df.groupby('gender').mean()
interests_df.describe()

##
interests_data = interests_df.iloc[:, 1:].to_numpy().T
interests_data = [col[~np.isnan(col)] for col in interests_data]


## plot
fig3, ax3 = plt.subplots(
    nrows=1, ncols=1, figsize=(8, 4), dpi=200,
    gridspec_kw={'left': 0.07, 'right': 0.96, 'top': 0.9, 'bottom': 0.14, 'hspace': 0.08})
fig3.suptitle("Figure 3: Distribution of interests among participants",
              x=0.02, ha='left')


vplot = ax3.violinplot(
    interests_data, positions=np.arange(1, 18), widths=0.6, showmeans=True)
for i, (m0, m1) in enumerate(mean_interests.T.to_numpy(), start=1):
    line_f, = ax3.plot([i-0.2, i+0.2], [m0, m0], color='tab:blue')
    line_m, = ax3.plot([i-0.2, i+0.2], [m1, m1], color='tab:orange')

ax3.set_xlim(0, 18)
ax3.set_xticks(np.arange(1, 18), interests, rotation=30)
ax3.set_ylim(0, 12)
ax3.set_ylabel('Score', labelpad=4)
ax3.grid(visible=True, axis='y', linewidth=0.3)
ax3.legend(handles=[line_f, line_m], labels=['Female', 'Male'], ncols=2)


for elt in vplot['bodies']:
    elt.set_facecolor('plum')
for elt in ['cbars', 'cmaxes', 'cmins', 'cmeans']:
    vplot[elt].set_edgecolor('plum')

plt.show()

"""
Choose violin plots because distrib bounded
We represent as violin plots since the values provided are bounded 
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
attributes = ['Attractive', 'Sincere', 'Intelligent', 'Fun', 'Ambitious', 'Shared\ninterests']
attr_kw = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']

##
cols = [[kw + f'{i}_{j}' for i in range(1, 6) for kw in attr_kw[:-1]]
        for j in range(1, 4)]
data = {'Before dating': subject_df.loc[:, cols[0]].count().to_numpy(),
        'After dating': subject_df.loc[:, cols[1]].count().to_numpy(),
        'Follow-up': subject_df.loc[:, cols[2]].count().to_numpy()}
df_ = pd.DataFrame.from_dict(data)
df_.index = [kw + f'{i}' for i in range(1, 6) for kw in attr_kw[:-1]]
df_

"""
Less responses over time
"""

#%%
"""
Focus on the self assessment and what I look for: same scale, 4 evals.

"""
##
ev_cols = [kw + f'3_{j}' for j in [1, 's', 2, 3] for kw in attr_kw[:-1]]
self_ev_df = subject_df.loc[:, ev_cols] # self evaluation dataframe
self_ev_df.count()

"""
We only consider the difference between after and before, since
there are less counts for the other situations, during and followup.
"""
##
# 4 sets of self assessed attributes
self_ev = np.array(np.split(self_ev_df.to_numpy(), 4, axis=1))
ev_diff = self_ev[2] - self_ev[0]
ev_diff = ev_diff[~np.any(np.isnan(ev_diff), axis=1)]


## do the same with what I look for
lf_cols = [kw + f'1_{j}' for j in [1, 's', 2, 3] for kw in attr_kw[:-1]]
self_lf_df = subject_df.loc[:, lf_cols] # self lookfor dataframe
self_lf_df.count()

self_lf = np.array(np.split(self_lf_df.to_numpy(), 4, axis=1))
self_lf = 100 * self_lf / np.sum(self_lf, axis=-1, keepdims=True)
lf_diff = self_lf[2] - self_lf[0]
lf_diff = lf_diff[~np.any(np.isnan(lf_diff), axis=1)]


##
fig4, axs4 = plt.subplots(
    nrows=2, ncols=5, figsize=(8, 6.4), dpi=200,
    gridspec_kw={'left': 0.08, 'right': 0.97, 'top': 0.89, 'bottom': 0.07,
                 'wspace': 0, 'hspace': 0.34})
fig4.suptitle("Figure 4: Variation of the self-evaluated attributes over the dating session",
              x=0.02, ha='left', fontsize=12)

for i, ax in enumerate(axs4[0]):
    ax.hist(ev_diff[:, i], bins=np.linspace(-5.5, 5.5, 12))
    ax.text(0.06, 0.83, textstr(ev_diff[:, i]),
            transform=ax.transAxes, fontsize=8,
            bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(0, 350)
    ax.grid(visible=True, axis='y', linewidth=0.3)
    ax.tick_params(left=False, labelleft=False)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel(attributes[i], labelpad=3)
axs4[0, 0].tick_params(left=True, labelleft=True)
axs4[0, 0].set_ylabel('Counts')
axs4[0, 0].set_title("Self-evaluated 'attr*_3' ('How do you think you measure up?')",
                     x=0, ha='left')


for i, ax in enumerate(axs4[1]):
    ax.hist(lf_diff[:, i], bins=np.linspace(-19.5, 19.5, 40))
    ax.text(0.06, 0.83, textstr(lf_diff[:, i]),
            transform=ax.transAxes, fontsize=8,
            bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    
    ax.set_xlim(-19, 19)
    ax.set_ylim(0, 120)
    ax.grid(visible=True, axis='y', linewidth=0.3)
    ax.tick_params(left=False, labelleft=False)
    ax.set_xticks([-10, 0, 10])
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15], minor=True)
    ax.set_xlabel(attributes[i], labelpad=3)
axs4[1, 0].tick_params(left=True, labelleft=True)
axs4[1, 0].set_ylabel('Counts')
axs4[1, 0].set_title("Self-evaluated 'attr*_1' ('What you look for in the opposite sex?')",
                     x=0, ha='left')


plt.show()

"""
No major change over time, maybe a slight decrease, indicating that people tend to
overestimate themselves.
"""


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


Here : how I perceive myself vs how ppl perceive me.
"""

# self-evauated attributes
sev_iids = self_ev_df.index.to_numpy()[~np.all(np.isnan(self_ev), axis=(0, 2))]
sev_genders = subject_df.loc[sev_iids]['gender'].to_numpy()
sev_attrs = self_ev[:, ~np.all(np.isnan(self_ev), axis=(0, 2))]
sev_attrs = np.mean(sev_attrs, axis=0, where=~np.isnan(sev_attrs))

# partner-evaluated attributes
pev_iids, pev_attrs = [], []
partner_ev_df = df.loc[:, ['pid'] + attr_kw[:-1]] # self evaluates partned
for iid, df_ in partner_ev_df.groupby('pid'):
    attr_eval = df_.loc[:, attr_kw[:-1]].to_numpy()
    mask = ~np.isnan(attr_eval)
    if np.all(np.any(mask, axis=0)):
        pev_iids.append(iid)
        pev_attrs.append(np.mean(attr_eval, axis=0, where=mask))
pev_iids, pev_attrs = np.array(pev_iids), np.array(pev_attrs)
pev_genders = subject_df.loc[pev_iids]['gender'].to_numpy()

# evaluation differences
sev_idx = np.isin(sev_iids, pev_iids, assume_unique=True)
pev_idx = np.isin(pev_iids, sev_iids, assume_unique=True)
ev_diff = sev_attrs[sev_idx] - pev_attrs[pev_idx]
ev_genders = sev_genders[sev_idx]


## plot
fig5, axs5 = plt.subplots(
    nrows=3, ncols=5, figsize=(7.5, 8.2), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.98, 'top': 0.88, 'bottom': 0.06,
                 'wspace': 0, 'hspace': 0.28})
fig5.suptitle("Figure 5: Differences between self-evaluated and partner-evaluated attributes",
              x=0.02, ha='left')

for i, ax in enumerate(axs5[0]):
    sev_f = sev_attrs[sev_genders==0, i]
    sev_m = sev_attrs[sev_genders==1, i]
    ax.violinplot(sev_f, positions=[-0.5], widths=0.8, showmeans=True)
    ax.violinplot(sev_m, positions=[0.5], widths=0.8, showmeans=True)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 12)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.grid(visible=True, axis='y', linewidth=0.3)
    ax.set_xlabel(attributes[i], labelpad=3)
    
axs5[0, 0].tick_params(bottom=False, left=True, labelleft=True)
axs5[0, 0].set_ylabel('Note')
axs5[0, 0].set_title("Self-evaluated 'attr*_3' ('How do you think you measure up?')",
                     x=0, ha='left')


for i, ax in enumerate(axs5[1]):
    pev_f = pev_attrs[pev_genders==0, i]
    pev_m = pev_attrs[pev_genders==1, i]
    ax.violinplot(pev_f, positions=[-0.5], widths=0.8, showmeans=True)
    ax.violinplot(pev_m, positions=[0.5], widths=0.8, showmeans=True)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 12)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.grid(visible=True, axis='y', linewidth=0.3)
    ax.set_xlabel(attributes[i], labelpad=3)
    
axs5[1, 0].tick_params(left=True, labelleft=True)
axs5[1, 0].set_ylabel('Note')
axs5[1, 0].set_title("Peer-evaluated 'attr' ('How do you think you measure up?')",
                     x=0, ha='left')


for i, ax in enumerate(axs5[2]):
    evd_f = ev_diff[ev_genders==0, i]
    evd_m = ev_diff[ev_genders==1, i]
    _, _, patch_f = ax.hist(evd_f, bins=np.linspace(-14.5, 14.5, 30),
                            alpha=0.6)
    _, _, patch_m = ax.hist(evd_m, bins=np.linspace(-14.5, 14.5, 30),
                            weights=np.full_like(evd_m, -1), alpha=0.6)
    ax.plot([np.mean(evd_f)]*2, [0, 170], color='k', linewidth=0.8)
    ax.plot([np.mean(evd_m)]*2, [0, -170], color='k', linewidth=0.8)
    ax.text(0.05, 0.8, textstr(evd_f), transform=ax.transAxes,
            fontsize=7, bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    ax.text(0.05, 0.03, textstr(evd_m), transform=ax.transAxes,
            fontsize=7, bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-170, 170)
    ax.grid(visible=True, axis='both', linewidth=0.3)
    ax.tick_params(left=False, labelleft=False)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_xlabel(attributes[i], labelpad=3)
    
axs5[2, 0].tick_params(left=True, labelleft=True)
axs5[2, 0].set_ylabel('Counts', labelpad=0)
axs5[2, 0].set_title("Difference between self evaluation and peer evaluation",
                     x=0, ha='left')

fig5.legend(handles=[patch_f, patch_m], labels=['Female', 'Male'],
            ncols=2, loc=(0.36, 0.915), alignment='center')

plt.show()


# TODO stat test for differences
"""

"""


#%%
"""
What I think I look for vs what I actually look for

The data is in some cases on a 1-10 scale, in other cases on a 100 pts total
scale. In this latter case, 


We compute the self-evaluated lookfor in the same way as the self evaluation of one'own attributes,
that is by averaging (after scaling) all the data available : before, during after, and at followup
For the real lookfor, we average the attributes of the people that the subject 'liked'.

Other's attribute and self not on the same scale...
!!!!! Correlation <attribute1_1,attr> 
"""

# TODO
# self-evauated attributes
# slf_iids = self_lf_df.index.to_numpy()[~np.all(np.isnan(self_lf), axis=(0, 2))]
# slf_genders = subject_df.loc[slf_iids]['gender'].to_numpy()
# slf_attrs = self_lf[:, ~np.all(np.isnan(self_ev), axis=(0, 2))]
# slf_attrs = np.mean(slf_attrs, axis=0, where=~np.isnan(slf_attrs))


# correlation matrices
gdf = df.groupby('gender')
lf_df_f = gdf.get_group(0).loc[:, [kw + '1_1' for kw in attr_kw] + attr_kw]
lf_corr_f = lf_df_f.corr().iloc[:6, 6:].to_numpy()
lf_df_m = gdf.get_group(1).loc[:, [kw + '1_1' for kw in attr_kw] + attr_kw]
lf_corr_m = lf_df_m.corr().iloc[:6, 6:].to_numpy()


# ## plot
fig6, axs6 = plt.subplots(
    nrows=1, ncols=2, figsize=(9, 4.6), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.9, 'top': 0.7, 'bottom': 0.16, 'wspace': 0.24})
cax6 = fig6.add_axes((0.92, 0.1, 0.025, 0.8))
fig6.suptitle("Figure 6: Differences between self-evaluated and partner-evaluated attributes",
              x=0.02, ha='left')


axs6[0].set_aspect('equal')
heatmap = axs6[0].pcolormesh(lf_corr_f[::-1]*100, cmap='coolwarm', vmin=-15, vmax=15,
                             edgecolors='0.2', linewidth=0.5)
axs6[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axs6[0].set_xticks(np.linspace(0.5, 5.5, 6), attributes, rotation=50)
axs6[0].set_yticks(np.linspace(0.5, 5.5, 6), attributes[::-1])
for (i, j), c in np.ndenumerate(lf_corr_f[::-1]*100):
    axs6[0].text(j+0.5, i+0.5, f'{c:.1f}', ha='center', va='center')

axs6[0].set_title('Females')
pos = axs6[0].get_position().bounds
x, y = pos[0], pos[1] + pos[3]
fig6.add_artist(Line2D([x-0.1, x], [y+0.1, y], color='k', lw=0.8))
fig6.text(x-0.02, y+0.09, 'eval. of\npartner', ha='center', va='center', fontweight='bold')
fig6.text(x-0.06, y, 'self\neval', ha='center', va='center', fontweight='bold')


axs6[1].set_aspect('equal')
heatmap = axs6[1].pcolormesh(lf_corr_m[::-1]*100, cmap='coolwarm', vmin=-15, vmax=15,
                             edgecolors='0.2', linewidth=0.5)
axs6[1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
axs6[1].set_xticks(np.linspace(0.5, 5.5, 6), attributes, rotation=50)
axs6[1].set_yticks(np.linspace(0.5, 5.5, 6), attributes[::-1])
for (i, j), c in np.ndenumerate(lf_corr_m[::-1]*100):
    axs6[1].text(j+0.5, i+0.5, f'{c:.1f}', ha='center', va='center')

axs6[1].set_title('Males')
pos = axs6[1].get_position().bounds
x, y = pos[0], pos[1] + pos[3]
fig6.add_artist(Line2D([x-0.1, x], [y+0.1, y], color='k', lw=0.8))
fig6.text(x-0.02, y+0.09, 'eval. of\npartner', ha='center', va='center', fontweight='bold')
fig6.text(x-0.06, y, 'self\neval', ha='center', va='center', fontweight='bold')


fig6.colorbar(heatmap, cax=cax6, orientation="vertical", ticklocation="right")
cax6.set_title('Correlation coef. (%)', y=-2.8)


plt.show()


##
"""

"""



# %%
"""
OK- Age diff in matches distrib
OK - Race match matrix
OK - nb likes vs each attr
OK - primality effect: nb likes vs 'order'
OK - matches vs 'int_corr'
OK - 'match_es' vs real number of matches
OK - 'expnum' vs actual nb of likes
"""

"""
Let us now explore the dating results proper.
"""

# TODO re-normalize ?
# TODO separate men and women


race_match_df = df.loc[df['gender']==0, ['match', 'race', 'race_o']]
norm = race_match_df.groupby(['race', 'race_o']).count()
matches = race_match_df.loc[df['match']==1].groupby(['race', 'race_o']).count()

match_idx = {k: i for i, k in enumerate(list(races.keys()))}
race_matches = np.full((len(races), len(races)), np.nan, dtype=float)
for (i, j), v in (matches/norm).iterrows():
    race_matches[match_idx[i], match_idx[j]] = v['match']


# race_match_df = df.loc[:, ['match', 'race', 'race_o']]
# norm = race_match_df.groupby(['race', 'race_o']).count()
# matches = race_match_df.loc[df['match']==1].groupby(['race', 'race_o']).count()

# match_idx = {k: i for i, k in enumerate(list(races.keys()))}
# race_matches = np.full((len(races), len(races)), np.nan, dtype=float)
# for (i, j), v in (matches/norm).iterrows():
#     race_matches[match_idx[i], match_idx[j]] = v['match']


## Plot heatmap
fig7, ax7 = plt.subplots(
    nrows=1, ncols=1, figsize=(4.5, 5), dpi=200,
    gridspec_kw={'left': 0.28, 'right': 0.94, 'top': 0.77, 'bottom': 0.17})
cax7 = fig7.add_axes((0.28, 0.1, 0.66, 0.035))
fig7.suptitle('Figure 7: Match probability vs race', x=0.02, ha='left')

ax7.set_aspect('equal')
cmap = plt.get_cmap('plasma').copy()
cmap.set_bad('0.6', 1)
heatmap = ax7.pcolormesh(race_matches[::-1], cmap=cmap, vmin=0, vmax=0.6,
                         edgecolors='0.2')
ax7.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax7.set_xticks(np.linspace(0.5, 5.5, 6), list(races.values()), rotation=50)
ax7.set_yticks(np.linspace(0.5, 5.5, 6), list(races.values())[::-1])
for (i, j), p in np.ndenumerate(race_matches[::-1]):
    if not np.isnan(p):
        ax7.text(j+0.5, i+0.5, f'{p:.2f}', ha='center', va='center')

fig7.colorbar(heatmap, cax=cax7, orientation="horizontal", ticklocation="bottom")
cax7.set_title('Match probability', y=-2.8)

pos = ax7.get_position().bounds
x, y = pos[0], pos[1] + pos[3]
fig7.add_artist(Line2D([x-0.14, x], [y+0.11, y], color='k', lw=0.8))
fig7.text(0.24, 0.89, 'Male\npartner', ha='center', va='center', fontweight='bold')
fig7.text(0.14, 0.81, 'Female\npartner', ha='center', va='center', fontweight='bold')


plt.show()

"""

"""



# %% 
"""
number of matches vs interests correlations
"""

int_corr = df['int_corr'].to_numpy()
match_int_corr = df.loc[df['match'] == 1, 'int_corr'].to_numpy()
bins = np.linspace(-1, 1, 41)

hist_ic = np.histogram(int_corr, bins=bins)[0]
hist_match_ic = np.histogram(match_int_corr, bins=bins)[0]

ic_vals = np.linspace(-0.975, 0.975, 40)
p_match = hist_match_ic / hist_ic # np.where(hist_ic, hist_ic, 1)
std_p_match = p_match * (1-p_match) / np.sqrt(np.where(hist_ic, hist_ic, 1))


## plot
fig8, axs8 = plt.subplots(
    nrows=1, ncols=2, figsize=(7, 3.5), dpi=200,
    gridspec_kw={'left': 0.09, 'right': 0.97, 'top': 0.88, 'bottom': 0.14, 'wspace':0.24})
fig8.suptitle('Figure 8: Effect of shared interests on matching', x=0.02, ha='left')


axs8[0].bar(ic_vals, hist_ic, width=0.05, label='all data')
axs8[0].bar(ic_vals, hist_match_ic, width=0.05, label='with match')
axs8[0].set_xlim(-1, 1)
axs8[0].set_ylim(0, 600)
axs8[0].grid(visible=True, linewidth=0.3)
axs8[0].set_xlabel('Interests correlation')
axs8[0].set_ylabel('Counts')
axs8[0].legend()

axs8[1].errorbar(ic_vals, p_match, yerr=std_p_match,
             fmt='o-', markersize=5)

axs8[1].set_xlim(-1, 1)
axs8[1].set_ylim(-0.005, 0.35)
axs8[1].set_yticks([0, 0.1, 0.2, 0.3])
axs8[1].set_yticks([0.05, 0.15, 0.25, 0.35], minor=True)
axs8[1].grid(visible=True, which='both', linewidth=0.3)
axs8[1].set_xlabel('Interests correlation')
axs8[1].set_ylabel('Match probability')

plt.show()

"""
There is a tendency for the match probability to increase with interests correlations.
However, the effect is very weak
"""


# %%
"""
Expected nb of likes / matches vs reality. for women and men

We do not study the number of likes: not enough values.
"""

iids = subject_df.index.to_numpy()
female_idx = np.array([i for i, iid in enumerate(iids)
                       if subject_df.loc[iid, 'gender'] == 0])
male_idx = np.array([i for i, iid in enumerate(iids)
                     if subject_df.loc[iid, 'gender'] == 1])

gdf = df.loc[:, ['iid', 'match', 'dec']].groupby('iid')
nmatch = np.array([gdf.get_group(i)['match'].sum() for i in iids])
exp_nmatch = subject_df.loc[iids, 'match_es'].to_numpy()


f_dmatch = (exp_nmatch - nmatch)[female_idx] # female match difference
m_dmatch = (exp_nmatch - nmatch)[male_idx]

## plot
fig9, ax9 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 3.5), dpi=200,
    gridspec_kw={'left': 0.12, 'right': 0.97, 'top': 0.88, 'bottom': 0.14})
fig9.suptitle('Figure 9: Matches, difference between expectations and reality',
              x=0.02, ha='left')

bins = np.linspace(-9.5, 9.5, 20)
ax9.hist(f_dmatch, bins=bins, label='Females')
ax9.plot([np.mean(f_dmatch, where=~np.isnan(f_dmatch))]*2, [0, 80],
             color='k', linewidth=0.8)
ax9.hist(m_dmatch, bins=bins, weights=np.full_like(m_dmatch, -1), label='Males')
ax9.plot([np.mean(m_dmatch, where=~np.isnan(m_dmatch))]*2, [0, -80],
             color='k', linewidth=0.8)
ax9.text(0.02, 0.825, textstr(f_dmatch), transform=ax9.transAxes,
        fontsize=9, bbox={'boxstyle': 'round', 'facecolor': '0.92'})
ax9.text(0.02, 0.04, textstr(m_dmatch), transform=ax9.transAxes,
        fontsize=9, bbox={'boxstyle': 'round', 'facecolor': '0.92'})

ax9.set_xlim(-8, 8)
ax9.set_ylim(-80, 80)
ax9.grid(visible=True, linewidth=0.3)
ax9.set_xlabel('Match difference (expectation - reality)')
ax9.set_ylabel('Counts')
ax9.legend()


plt.show()

##
# TODO stat test

"""


"""



# %%
"""
Age and income difference distribution when match
"""

gd_df = df.loc[df['match']==1, ['pid', 'gender', 'age', 'income']]
gd_df['age_o'] = [subject_df.loc[int(pid), 'age'] for pid in gd_df['pid']]
gd_df['income_o'] = [subject_df.loc[int(pid), 'income'] for pid in gd_df['pid']]

gd_df = gd_df.loc[gd_df['gender']==0]
age_diff = (gd_df['age'] - gd_df['age_o']).to_numpy() # female - male
income_diff = (gd_df['income'] - gd_df['income_o']).to_numpy() 


## plot
fig10, axs10 = plt.subplots(
    nrows=1, ncols=2, figsize=(7.2, 3.5), dpi=200,
    gridspec_kw={'left': 0.08, 'right': 0.97, 'top': 0.88, 'bottom': 0.14, 'wspace':0.24})
fig10.suptitle('Figure 10: Age and income difference in matches (females - males)',
               x=0.02, ha='left')


axs10[0].hist(age_diff, bins=np.linspace(-14.5, 14.5, 30))
axs10[0].plot([np.mean(age_diff, where=~np.isnan(age_diff))]*2, [0, 80],
              color='k', linewidth=0.8)
axs10[0].text(0.03, 0.825, textstr(age_diff), transform=axs10[0].transAxes,
              fontsize=9, bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs10[0].set_xlim(-15, 15)
axs10[0].set_ylim(0, 80)
axs10[0].grid(visible=True, linewidth=0.3)
axs10[0].set_xlabel('Age difference (years)')
axs10[0].set_ylabel('Counts')
# axs10[0].legend()


axs10[1].hist(income_diff, bins=np.linspace(-65e3, 65e3, 27))
axs10[1].plot([np.mean(income_diff, where=~np.isnan(income_diff))]*2, [0, 30],
              color='k', linewidth=0.8)
axs10[1].text(0.03, 0.825, textstr(income_diff/1e3), transform=axs10[1].transAxes,
              fontsize=9, bbox={'boxstyle': 'round', 'facecolor': '0.92'})

axs10[1].set_xlim(-80e3, 80e3)
axs10[1].set_ylim(0, 30)
axs10[1].set_xticks(np.linspace(-80e3, 80e3, 9), np.arange(-80, 100, 20))
axs10[1].grid(visible=True, which='both', linewidth=0.3)
axs10[1].set_xlabel('Income difference (k$)')
axs10[1].set_ylabel('Counts')

plt.show()


# TODO stat test
"""

"""



# %%

"""
Attribute difference when like

"""

attr_data = df.loc[df['dec']==1, ['gender'] + [kw + '3_1' for kw in attr_kw[:-1]] + attr_kw[:-1]]
attr_data = attr_data.astype(float).to_numpy()

bins = np.linspace(-10.5, 10.5, 22)
x = np.linspace(-10, 10, 21)

f_attr = attr_data[attr_data[:, 0]==0, 1:]
f_attr_diff = f_attr[:, 5:] - f_attr[:, :5] # self eval - other eval
f_attr_hist = [np.histogram(d, bins=bins)[0] for d in f_attr_diff.T]
f_attr_mean = np.mean(f_attr_diff, axis=0, where=~np.isnan(f_attr_diff))

m_attr = attr_data[attr_data[:, 0]==1, 1:]
m_attr_diff = m_attr[:, 5:] - m_attr[:, :5]
m_attr_hist = [np.histogram(d, bins=bins)[0] for d in m_attr_diff.T]
m_attr_mean = np.mean(m_attr_diff, axis=0, where=~np.isnan(m_attr_diff))


## plot
fig11, axs11 = plt.subplots(
    nrows=1, ncols=5, figsize=(9, 3.8), dpi=200, sharex=True, sharey=True,
    gridspec_kw={'left': 0.08, 'right': 0.975, 'top': 0.85, 'bottom': 0.14, 'wspace': 0})
fig11.suptitle('Figure 11: Distribution of the attribute rating difference when "liking" a partner',
               x=0.02, ha='left')

for i, ax in enumerate(axs11):
    bar_f = ax.bar(x, f_attr_hist[i]/np.sum(f_attr_hist[i]), width=1)
    ax.plot([f_attr_mean[i]]*2, [0, 0.3], color='k', linewidth=0.6)
    ax.text(0.05, 0.85, textstr(f_attr_diff[:, i]), transform=ax.transAxes,
            fontsize=8, bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    bar_m = ax.bar(x, -m_attr_hist[i]/np.sum(m_attr_hist[i]), width=1)
    ax.plot([m_attr_mean[i]]*2, [0, -0.3], color='k', linewidth=0.6)
    ax.text(0.05, 0.03, textstr(m_attr_diff[:, i]), transform=ax.transAxes,
            fontsize=8, bbox={'boxstyle': 'round', 'facecolor': '0.92'})
    
    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-8, 10, 4))
    ax.set_xticks(np.arange(-10, 12, 4), minor=True)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks(np.linspace(-0.35, 0.35, 8), minor=True)
    ax.grid(visible=True, which='both', linewidth=0.3)
    ax.set_xlabel(attributes[i])

axs11[0].set_ylabel('Probability')
fig11.legend(handles=[bar_f, bar_m], labels=['Female', 'Male'],
             ncols=2, loc=(0.36, 0.86), alignment='center')

plt.show()

# TODO stat test
"""

"""



# %%
"""
number of likes vs dating order
"""

order_counts = df['order'].value_counts().to_numpy()
likes_vs_order = df['order'].loc[df['dec'] == 1].value_counts()

like_prob_df = df.loc[:, ['order', 'dec']].groupby('order').mean()
order = like_prob_df.index.to_numpy()
like_prob = like_prob_df['dec'].to_numpy() / 2
like_std = like_prob * (1 - like_prob) / np.sqrt(order_counts)


## plot
fig12, ax12 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 3.5), dpi=200,
    gridspec_kw={'left': 0.12, 'right': 0.97, 'top': 0.88, 'bottom': 0.14})
fig12.suptitle('Figure 12: Primality effect', x=0.02, ha='left')

ax12.errorbar(order, like_prob, like_std, fmt='o-')

ax12.set_xlim(0, 23)
ax12.set_ylim(0.14, 0.28)
ax12.grid(visible=True, linewidth=0.3)
ax12.set_xlabel('Date order')
ax12.set_ylabel('Like probability')

plt.show()

"""

"""
