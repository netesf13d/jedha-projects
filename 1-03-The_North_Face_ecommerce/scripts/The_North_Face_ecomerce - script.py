# -*- coding: utf-8 -*-
"""
Script version of the North Face ecommerce project.
"""

import csv
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import wordcloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# from spacy.lang.en.stop_words import STOP_WORDS
# from spacy.tokens.doc import Doc


# %% Loading and preprocessing
"""
## <a id="loading"></a> Data loading and preprocessing

### Preparation of the corpus
"""

## Load the file
with open('../sample-data.csv', 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_STRINGS)
    next(reader, None) # remove header
    data = [row for row in reader]

description_id = [row[0] for row in data]
descriptions = [row[1] for row in data]


## remove HTML tags
descriptions = [re.sub(r'<[a-z/]+>', ' ', d) for d in descriptions]

# download with `python -m spacy download en_core_web_sm`
## normalize into lemmas
nlp = spacy.load('en_core_web_sm')
corpus = []
for desc in descriptions:
    doc = nlp(desc)
    doc = ' '.join(wd.lemma_.lower() for wd in doc
                   if not wd.is_stop and (wd.is_alpha))#  or wd.like_num))
    corpus.append(doc)
corpus = np.array(corpus, dtype=object)


#%% TF-IDF
"""
### Construction of the TF-IDF matrix

"""
# vocab, vocab_counts = np.unique(np.concatenate(lemma_corpus), return_counts=True)
vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
word_counts = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()

tfidf = TfidfTransformer(norm='l2').fit_transform(word_counts)


#%%
"""
### A first wordcloud

"""


# def create_wordcloud(corpus: list[str])-> np.ndarray:
#     """
#     !!! re-doc
#     Return a numpy array representing a wordcloud of the array.
#     """
#     word_count = Counter(corpus)
#     wc = wordcloud.WordCloud(width=1000, height=500, min_font_size=14,
#                              colormap='rainbow', random_state=1234)
#     wc = wc.generate_from_frequencies(word_count)
#     return wc.to_array()
    
def create_wordcloud(text: str, **kwargs)-> np.ndarray:
    """
    !!! re-doc
    Return a numpy array representing a wordcloud of the array.
    """
    defaults = {'width': 1000, 'height': 500, 'min_font_size': 14,
                'colormap': 'rainbow', 'random_state': 1234}    
    wc = wordcloud.WordCloud(**(defaults | kwargs))
    wc = wc.generate(text)
    return wc.to_array()



# word_count = Counter(np.concatenate(lemma_corpus))
# wc = wordcloud.WordCloud(width=1000, height=500, min_font_size=16,
#                          colormap='rainbow')
# wc = wc.generate_from_frequencies(word_count)
# im = wc.to_image()

wc_arr = create_wordcloud(np.concatenate(corpus))
xy_ratio = wc_arr.shape[0] / wc_arr.shape[1]

wc_fig1 = plt.figure(figsize=(6, 6*xy_ratio), dpi=100)
wc_ax1 = wc_fig1.add_axes(rect=(0, 0, 1, 1))

wc_ax1.axis('off')
wc_ax1.imshow(wc_arr, interpolation='bilinear')

plt.show()


# import displacy
from spacy import displacy

# render displacy
displacy.render(doc, style='dep', jupyter=True,options={'distance': 70})



# %% clustering
"""
## <a id="clustering"></a> Descriptions clustering

Our data lies in high dimension (about 4000), with only 500 observations.

### Clustering with DBSCAN

"""



dbc = DBSCAN(eps=0.6, min_samples=8, metric='cosine').fit(tfidf)
dbc_labels = dbc.labels_
nb_dbc = len(np.unique(dbc_labels)) - 1

print('======== DBSCAN ========',
      f'\nNumber of clusters: {nb_dbc}',
      f'\nNumber of outliers: {np.sum(dbc_labels==-1)}')


#%%
"""
### Clustering with HDBSCAN

"""

hdbc = HDBSCAN(metric='cosine').fit(tfidf)
hdbc_labels = hdbc.labels_
nb_hdbc = len(np.unique(hdbc_labels)) - 1

print('======== HDBSCAN ========',
      f'\nNumber of clusters: {nb_hdbc}',
      f'\nNumber of outliers: {np.sum(hdbc_labels==-1)}')


#%%
"""
### Displaying the results

"""

tsne = TSNE(metric='cosine', random_state=0, init='random')
embedded_tfidf = tsne.fit_transform(tfidf)


#%%

COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]

##
fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4.8), dpi=100,
    gridspec_kw={'left': 0.04, 'right': 0.96, 'top': 0.72, 'bottom': 0.06, 'wspace': 0.1})
fig1.suptitle("Figure 1: Time dependence of CPI and temperature",
              x=0.02, ha='left', fontsize=12)

# handles, labels = [], []
# for istore, store_df in df.groupby('Store'):
#     # CPI vs date
#     line, = axs1[0].plot(store_df['Date'], store_df['CPI'], color=COLORS[istore],
#                          marker='o', markersize=4, linestyle='')

#     # temperature vs month
#     gdf = store_df.groupby('Month').mean()
#     axs1[1].plot(gdf.index, gdf['Temperature'], color=COLORS[istore],
#                  marker='o', markersize=5, linestyle='')

#     # legend handles and labels
#     handles.append(line)
#     labels.append(f'store {istore}')

# display outliers
outlier_idx = np.nonzero(dbc_labels == -1)[0]
axs1[0].plot(embedded_tfidf[outlier_idx, 0], embedded_tfidf[outlier_idx, 1],
             marker='.', linestyle='', color='k')
# display clusters
for k in range(nb_dbc):
    idx = np.nonzero(dbc_labels == k)[0]
    axs1[0].plot(embedded_tfidf[idx, 0], embedded_tfidf[idx, 1],
                 marker='.', linestyle='', color=COLORS[k])


axs1[0].tick_params(direction='in', labelleft=False, right=True,
                    labelbottom=False, top=True)
axs1[0].grid(visible=True, linewidth=0.2)
axs1[0].set_xlim(-30, 30)
axs1[0].set_xticks(np.linspace(-30, 30, 7))
axs1[0].set_ylim(-50, 50)
axs1[0].set_yticks(np.linspace(-40, 40, 5))
axs1[0].set_title('DBSCAN clustering')


# display outliers
outlier_idx = np.nonzero(hdbc_labels == -1)[0]
axs1[1].plot(embedded_tfidf[outlier_idx, 0], embedded_tfidf[outlier_idx, 1],
             marker='.', linestyle='', color='k')
# display clusters
for k in range(nb_hdbc):
    idx = np.nonzero(hdbc_labels == k)[0]
    axs1[1].plot(embedded_tfidf[idx, 0], embedded_tfidf[idx, 1],
                 marker='.', linestyle='', color=COLORS[k])

axs1[1].tick_params(direction='in', labelleft=False, right=True,
                    labelbottom=False, top=True)
axs1[1].grid(visible=True, linewidth=0.2)
axs1[1].set_title('HDBSCAN clustering')

# fig1.legend(handles=handles, labels=labels,
#             ncols=5, loc=(0.14, 0.745), alignment='center')

plt.show()


#%%

"""
We can also display wordclouds associated to the various clusters.
"""


fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, sharey=True, figsize=(7, 8), dpi=100,
    gridspec_kw={'left': 0.09, 'right': 0.96, 'top': 0.83, 'bottom': 0.065,
                 'wspace': 0.10, 'hspace': 0.3})
fig2.suptitle("Figure 3: Scatter plots of the weekly sales vs the various quantitative features",
              x=0.02, y=0.99, ha='left')


for k, ax in enumerate(axs2.ravel()):
    idx = np.nonzero(hdbc_labels == k)[0]
    text = ' '.join(corpus[i] for i in idx)
    ax.imshow(create_wordcloud(text, min_font_size=20),
              interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'cluster {k}')


plt.show()




# %% recommender system
"""
## <a id="recommendation"></a> Recommender system

"""

labels = hdbc_labels
id_to_cluster = {id_: labels[i] for i, id_ in enumerate(description_id)}

def find_similar_items(item_id: int, n_items: int=5):
    
    if item_id not in id_to_cluster:
        raise KeyError(f'no item with id={item_id}')
    
    cluster_idx = np.nonzero(labels == id_to_cluster[item_id])[0]
    
    i0 = description_id.index(item_id)
    similarities = [np.dot(tfidf[i0], tfidf[i]) for i in cluster_idx]
    
    # indices most similar items sorted by decreasing similarity
    best_items_idx = cluster_idx[np.argsort(similarities)[::-1]]
    
    return corpus[best_items_idx[1:1+n_items]]






# %% topic modeling
"""
## <a id="topic"></a> Topic modeling

"""

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd = svd.fit(tfidf)

