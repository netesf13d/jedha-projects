#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Your beloved one
"""

from io import BytesIO


import numpy as np
import pandas as pd


# =============================================================================
# 
# =============================================================================

fname = "./Speed+Dating+Data.csv"
df = pd.read_csv(fname, encoding='mac_roman')
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
cols = ['age', 'field', 'career', 'goal']
print(subject_df[cols].to_numpy())

