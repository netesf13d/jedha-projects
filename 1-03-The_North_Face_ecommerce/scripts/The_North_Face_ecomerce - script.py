# -*- coding: utf-8 -*-
"""
Script version of the North Face ecommerce project.
"""

import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import wordcloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD



# %% Loading and preprocessing
# !!!
"""
## <a id="loading"></a> Data loading and preprocessing

### Preparation of the corpus
"""

## Load the file
with open('../sample-data.csv', 'rt', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_STRINGS)
    next(reader, None) # remove header
    data = [row for row in reader]

description_id = [int(row[0]) for row in data]
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
# !!!
"""
### Construction of the TF-IDF matrix

"""
# vocab, vocab_counts = np.unique(np.concatenate(lemma_corpus), return_counts=True)
vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
word_counts = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()

tfidf = TfidfTransformer(norm='l2').fit_transform(word_counts)


#%%
# !!!
"""
### A first wordcloud

"""
    
def create_wordcloud(text_or_freqs: str | dict[str, float],
                     **kwargs)-> np.ndarray:
    """
    !!! re-doc
    Return a numpy array representing a wordcloud of the array.
    """
    defaults = {'width': 1000, 'height': 500, 'min_font_size': 14,
                'colormap': 'gist_rainbow', 'collocations': False,
                'random_state': 1234}    
    wc = wordcloud.WordCloud(**(defaults | kwargs))
    if isinstance(text_or_freqs, str):
        wc = wc.generate(text_or_freqs)
    elif isinstance(text_or_freqs, dict):
        wc = wc.generate_from_frequencies(text_or_freqs)
    else:
        raise TypeError('`text_or_freqs` must be of type `str` or `dict`')
    return wc.to_array()


##
wc_arr = create_wordcloud(' '.join(corpus))
xy_ratio = wc_arr.shape[0] / wc_arr.shape[1]

wc_fig1 = plt.figure(figsize=(6, 6*xy_ratio), dpi=100)
wc_ax1 = wc_fig1.add_axes(rect=(0, 0, 1, 1))

wc_ax1.axis('off')
wc_ax1.imshow(wc_arr, interpolation='bilinear')

plt.show()

#!!!
"""

"""


# %% clustering
#!!!
"""
## <a id="clustering"></a> Descriptions clustering

Our data lies in high dimension (about 4000), with only 500 observations.

### Clustering with DBSCAN

"""
## Setup scanned parameters
eps_vals = np.linspace(0.5, 0.9, 21)
min_samples_vals = np.arange(1, 16, 1)
sh = (len(min_samples_vals), len(eps_vals))

## Grid search and record number of clusters and outliers
nb_clusters = np.zeros(sh, dtype=int)
nb_outliers = np.zeros(sh, dtype=int)
for i, j in np.ndindex(sh):
    dbc = DBSCAN(eps=eps_vals[j], min_samples=min_samples_vals[i],
                 metric='cosine', leaf_size=1)
    dbc = dbc.fit(tfidf)
    dbc_labels = dbc.labels_
    nb_clusters[i, j] = len(np.unique(dbc_labels)) - 1
    nb_outliers[i, j] = np.sum(dbc_labels==-1)
    

# %%
##
fig1, axs1 = plt.subplots(
    nrows=2, ncols=2, sharex=False, sharey=False, figsize=(8.4, 4.2), dpi=200,
    gridspec_kw={'left': 0.07, 'right': 0.96, 'top': 0.81, 'bottom': 0.09,
                 'height_ratios': [0.07, 1], 'hspace': 0.04, 'wspace': 0.12})
fig1.suptitle("Figure 1: Maps of numbers of clusters and outliers for the grid search",
              x=0.02, ha='left', fontsize=12)


axs1[1, 0].set_aspect('equal')
cmap1 = plt.get_cmap('tab20b').copy()
cmap1.set_extremes(under='0.1', over='0.9')
heatmap1 = axs1[1, 0].pcolormesh(nb_clusters, cmap=cmap1, vmin=4.5, vmax=24.5,
                                 edgecolors='0.15', linewidth=0.05)

axs1[1, 0].set_xticks(np.linspace(0.5, 20.5, 5), eps_vals[::5])
axs1[1, 0].set_yticks(np.linspace(0.5, 14.5, 8), min_samples_vals[::2])
axs1[1, 0].set_ylabel('`min_samples`', fontsize=11)
fig1.text(0.5, 0.022, 'max sample distance (`eps`)', fontsize=11, ha='center')

fig1.colorbar(heatmap1, cax=axs1[0, 0], orientation='horizontal', ticklocation='top')
axs1[0, 0].tick_params(pad=1)
axs1[0, 0].set_xticks(np.linspace(5, 23, 10), np.arange(5, 24, 2))
axs1[0, 0].set_title('Number of clusters')


axs1[1, 1].set_aspect('equal')
cmap2 = plt.get_cmap('plasma').copy()
cmap2.set_extremes(under='0.1', over='0.9')
heatmap2 = axs1[1, 1].pcolormesh(nb_outliers, cmap=cmap2, vmin=0, vmax=100,
                                 edgecolors='0.15', linewidth=0.1)

axs1[1, 1].set_xticks(np.linspace(0.5, 20.5, 5), eps_vals[::5])
axs1[1, 1].set_yticks(np.linspace(0.5, 14.5, 8), min_samples_vals[::2])

fig1.colorbar(heatmap2, cax=axs1[0, 1], orientation='horizontal', ticklocation='top')
axs1[0, 1].tick_params(pad=1)
axs1[0, 1].set_title('Number of outliers')


plt.show()

# !!!
"""
Figure 1 is so nice

"""

#%%


valid_params_idx = (nb_outliers <= 60) * (nb_clusters >= 8) * (nb_clusters <= 24)
valid_min_samples_idx, valid_eps_idx = np.nonzero(valid_params_idx)

for i, j in zip(valid_min_samples_idx, valid_eps_idx):
    print(f'min_samples = {min_samples_vals[i]} ; eps = {eps_vals[j]:<5.2f}',
          f'==>  {nb_clusters[i, j]:>2} clusters, {nb_outliers[i, j]:>2} outliers')

# !!!
"""
We retrain the configuration with 9 clusters and 7 outliers.
"""


# %%

dbc = DBSCAN(eps=0.7, min_samples=4, metric='cosine', leaf_size=1).fit(tfidf)
dbc_labels = dbc.labels_
nb_dbc = len(np.unique(dbc_labels)) - 1

print('======== DBSCAN ========',
      f'\nNumber of clusters: {nb_dbc}',
      f'\nNumber of outliers: {np.sum(dbc_labels==-1)}')
print(np.unique(dbc_labels, return_counts=True)[0])
print(np.unique(dbc_labels, return_counts=True)[1])


#%%
# !!!
"""
### Clustering with HDBSCAN

"""

hdbc = HDBSCAN(alpha=1.15, cluster_selection_epsilon=0.6,
               metric='cosine', leaf_size=1)
hdbc = hdbc.fit(tfidf)
hdbc_labels = hdbc.labels_
nb_hdbc = len(np.unique(hdbc_labels)) - 1

print('======== HDBSCAN ========',
      f'\nNumber of clusters: {nb_hdbc}',
      f'\nNumber of outliers: {np.sum(hdbc_labels==-1)}')
print(np.unique(hdbc_labels, return_counts=True)[0])
print(np.unique(hdbc_labels, return_counts=True)[1])


#%%
# !!!
"""
### Displaying the results

"""

tsne = TSNE(perplexity=10, early_exaggeration=20, learning_rate=400,
            metric='cosine', random_state=0, init='random')
embedded_tfidf = tsne.fit_transform(tfidf)


#%%

## reindex
reindex = [6, 9, 15, 3, 11, 2, 4, 0, 7, 8, 14, 5, 12, 1, 10]
new_hdbc_labels = np.full_like(hdbc_labels, fill_value=-1)
for i, k in enumerate(reindex):
    new_hdbc_labels[np.nonzero(hdbc_labels == i)] = k


COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]

##
fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4.8), dpi=100,
    gridspec_kw={'left': 0.04, 'right': 0.96, 'top': 0.74, 'bottom': 0.07, 'wspace': 0.1})
fig2.suptitle("Figure 2: t-SNE representation of the clusterized descriptions",
              x=0.02, ha='left', fontsize=12)


# display outliers
idx = np.nonzero(dbc_labels == -1)[0]
line, = axs2[0].plot(embedded_tfidf[idx, 0], embedded_tfidf[idx, 1],
                     marker='.', linestyle='', color='k')
handles, labels = [line], ['outliers']

# display clusters
for k in range(nb_dbc):
    idx = np.nonzero(dbc_labels == k)[0]
    line, = axs2[0].plot(embedded_tfidf[idx, 0], embedded_tfidf[idx, 1],
                         marker='.', linestyle='', color=COLORS[k])
    handles.append(line)
    labels.append(f'cluster {k}')


axs2[0].tick_params(direction='in', labelleft=False, right=True,
                    labelbottom=False, top=True)
axs2[0].grid(visible=True, linewidth=0.2)
axs2[0].set_xlim(-70, 70)
# axs2[0].set_xticks(np.linspace(-30, 30, 7))
axs2[0].set_ylim(-75, 75)
# axs2[0].set_yticks(np.linspace(-75, 75, 5))
axs2[0].set_title(f'DBSCAN ({nb_dbc-1} clusters)', y=-0.1)


# display outliers
outlier_idx = np.nonzero(hdbc_labels == -1)[0]
axs2[1].plot(embedded_tfidf[outlier_idx, 0], embedded_tfidf[outlier_idx, 1],
             marker='.', linestyle='', color='k')
# display clusters
for k in range(nb_dbc):
    idx = np.nonzero(new_hdbc_labels == k)[0]
    axs2[1].plot(embedded_tfidf[idx, 0], embedded_tfidf[idx, 1],
                 marker='.', linestyle='', color=COLORS[k])

axs2[1].tick_params(direction='in', labelleft=False, right=True,
                    labelbottom=False, top=True)
axs2[1].grid(visible=True, linewidth=0.2)
axs2[1].set_title(f'HDBSCAN ({nb_hdbc-1} clusters)', y=-0.1)

fig2.legend(handles=handles, labels=labels,
            ncols=6, loc=(0.016, 0.77), alignment='center')

plt.show()

# !!!
"""

There is no cluster 13 on HDBSCAN
"""



#%%
# !!!
"""
We can also display word clouds associated to the various clusters.
"""

mask = np.full((500, 1000), 0x0, dtype=int)
mask[420:, :120] = 0xff

clusters_wc = []
for k in range(-1, nb_dbc):
    text = ' '.join(corpus[np.nonzero(dbc_labels == k)])
    wc = create_wordcloud(text, min_font_size=20, mask=mask)
    clusters_wc.append(wc)
cluster_names = [f'{i}' for i in range(-1, nb_dbc)]


# %%

fig3, axs3 = plt.subplots(
    nrows=8, ncols=2, sharey=True, figsize=(5, 10.4), dpi=100,
    gridspec_kw={'left': 0.03, 'right': 0.97, 'top': 0.97, 'bottom': 0.01,
                 'wspace': 0.06, 'hspace': 0.1})
fig3.suptitle("Figure 3: nice wordclouds",
              x=0.02, y=0.99, ha='left')


for k, ax in enumerate(axs3.ravel()):
    ax.axis('off')
    ax.imshow(clusters_wc[k], interpolation='bilinear')
    ax.text(0.05, 0.05, cluster_names[k], color='w',
            fontsize=10, fontweight='bold', ha='center',
            transform=ax.transAxes)

plt.show()

# !!!
"""
Figure 3
"""


# %% recommender system
# !!!
"""
## <a id="recommendation"></a> Recommender system

"""

descriptions_df = pd.DataFrame({'description': descriptions}, index=description_id)


def find_similar_items(item_id: int, n_items: int = 5)-> pd.DataFrame:
    
    if item_id not in descriptions_df.index:
        raise KeyError(f'no item with id={item_id}')
    
    i0 = description_id.index(item_id)
    similarities = (tfidf @ tfidf[i0].T).toarray().ravel()
    
    # indices most similar items sorted by decreasing similarity
    best_items_idx = np.argsort(similarities)[::-1]
    
    return descriptions_df.iloc[best_items_idx[1:n_items+1], 0]


## 
descriptions_df.loc[1, 'description']

##
find_similar_items(1, n_items=3)

# !!!
"""

"""


# %% topic modeling 
"""
## <a id="topic"></a> Topic modeling

topics are sorted by decreasing imporance in the corpus.
"""

## compute only the 10 most important topics
svd = TruncatedSVD(n_components=10, random_state=1234)
svd = svd.fit(tfidf)

topics = svd.components_


## 
topic_decomp_df = pd.DataFrame(
    tfidf @ topics.T, index=description_id,
    columns=[f'topic {i}' for i in range(1, 11)])


# %%
"""
Let us visualize the topics
"""

## Positive coefficients wordclouds
topics_pos = [dict(zip(vocab[topic > 0], topic[topic > 0]))
              for topic in topics]
topic_wordclouds_pos = [create_wordcloud(t, min_font_size=25) if t else None
                        for t in topics_pos]

## Negative coefficients wordclouds
topics_neg = [dict(zip(vocab[topic < 0], topic[topic < 0]))
              for topic in topics]
topic_wordclouds_neg = [create_wordcloud(t, min_font_size=25) if t else None
                        for t in topics_neg]


#%%

fig4, axs4 = plt.subplots(
    nrows=5, ncols=2, sharey=True, figsize=(5.5, 7.5), dpi=100,
    gridspec_kw={'left': 0.06, 'right': 0.97, 'top': 0.95, 'bottom': 0.04,
                 'wspace': 0.05, 'hspace': 0.1})
fig4.suptitle("Figure 4: Even nicer wordclouds", fontweight='normal',
              x=0.02, y=0.99, ha='left')

axs4[-1, 0].text(0.5, -0.14, 'Positive component', fontsize=11, fontweight='bold',
                transform=axs4[-1, 0].transAxes, ha='center', va='center')
axs4[-1, 1].text(0.5, -0.14, 'Negative component', fontsize=11, fontweight='bold',
                transform=axs4[-1, 1].transAxes, ha='center', va='center')

for k, axs in enumerate(axs4):
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].text(-0.06, 0.5, f'Topic {k+1}', fontsize=11, fontweight='bold',
                transform=axs[0].transAxes, ha='center', va='center', rotation=90)
    
    if topic_wordclouds_pos[k] is not None:
        axs[0].imshow(topic_wordclouds_pos[k], interpolation='bilinear')
    if topic_wordclouds_neg[k] is not None:
        axs[1].imshow(topic_wordclouds_neg[k], interpolation='bilinear')

plt.show()

"""
Figure 4
"""


# %% Conclusion

"""
## <a id="conclusion"></a> Conclusion

Should split the descriptions
"""


