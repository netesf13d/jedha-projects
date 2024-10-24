#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Your beloved one
"""

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
# with open(fname, 'rb') as f:
#     raw_data = f.read()
# raw_data.replace(b'\x8ee', 'é'.encode('utf-8'))

# for col in df:
#     if df[col].dtype == object:
#         print(df[col].str.contains("assistant master"))
        # if any(df[col].str.contains("Ecole Normale")):
        #     print(col, df[col])

# troll_entry = df[df['career'].str.contains("assistant master")==True]
# select = ~troll_entry.isnull().any().to_numpy()
# # troll_entry = troll_entry.iloc[:, select]
# a = troll_entry[['wave', 'field_cd', 'age', 'exercise', 'zipcode', 'career_c']]
# print(a.head())

subject_entries = pd.concat([pd.Series([True], index=['iid']),
                             df.groupby('iid').nunique().max() <= 1])
partner_entries = ~subject_entries


subject_cols = df.columns[subject_entries]
subject_df = df[subject_cols].groupby('iid').aggregate(lambda x: x.iloc[0])

#%%

cols = ['age', 'income', 'mn_sat']
for c in cols:
    if sum(~np.isnan(subject_df[c])) > 276:
        print(c)


#%%

ages = subject_df['age'].to_numpy()
ages = ages[~np.isnan(ages)]
bins = np.arange(10, 61, 1)

mn_sat = subject_df['mn_sat']
income = subject_df['income']

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].hist(subject_df['age'], bins=np.linspace(10, 60, 51))
ax[0].set_xlim(10, 60)
ax[0].set_title("Age distribution")

ax[1].hist(subject_df['income'], bins=np.linspace(1e4, 1.2e5, 23))
ax[1].set_xlim(0, 1.2e5)
ax[1].set_title("Income distribution")

plt.show()


#%%

['sports', 'tvsports', 'exercise', 'dining',
'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga']






