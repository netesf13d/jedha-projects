# -*- coding: utf-8 -*-
"""
Script for the AT&T spam detector project.
"""


import sys
import time
import re
import unicodedata
from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras


SEED = 1234
rng = np.random.default_rng(seed=SEED)


# %% Loading
r'''
## <a id="loading"></a> Data loading and preprocessing

The dataset consists of 5574 entries, many of which have issues. Here is a non-exhaustive list:
- The file doesn't decode as `UTF-8`. It must be decoded as `latin-1` or `mac-roman`.
- Some rows are ill-formed (eg line 101 of the csv: `ham""",,,`)
- Most rows have trailing commas
- Some messages are anclosed in (possibly many) double quote characters
- Single and double quotes are escaped with backslashes
- Some spurious characters are present (eg `å` characters that systematically prepend pound symbols `£`)
- The messages contain some HTML escaped characters (eg `'&gt;'` or `'&amp;'`)

Fortunately, the original version of the dataset can be found
[here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Its entries are much cleaner
than those of our dataset. We can use them as a comparison to setup the cleaning and processing for our dataset:
- Strip the trailing commas
- Completely remove the double quotes `"`
- Convert backslashes + single quote `\'` into single quote `'`
- Convert backslashes `\` into double quotes `"`
- Remove the spurious characters `å`
- Replace the HTML entities with their corresponding symbols

The implementation of this pipeline is given below.
'''

# ########## Loading and preprocessing the project dataset ##########
# with open('../spam.csv', 'rt', encoding='latin-1') as f:
#     f.readline() # remove header
#     raw_data = [row.strip().split(',', 1) for row in f.readlines()]

# ## set target
# is_spam = np.array([True if row[0] == 'spam' else False for row in raw_data])

# ## process and clean messages
# messages = [row[1].strip(',') \
#                   .replace('"', '') \
#                   .replace("\\'", "'") \
#                   .replace('\\', '"') \
#                   .replace('å', '') \
#                   .replace('&lt;', '<') \
#                   .replace('&gt;', '>') \
#                   .replace('&amp;', '&')
#             for row in raw_data]


"""
However, some encoding issues persist with some problematic characters still present in messages.
To proceed with the project we will rather use this original dataset as the starting point.
The messages still need some processing:
- characters in HTML-escaped form are converted to the corresponding symbol
- The `RIGHT SINGLE QUOTATION MARK` character (`\x92`) is converted into a single quote `'`
"""

########## Loading and preprocessing the original dataset ##########
with open('../SMSSpamCollection', 'rt', encoding='utf-8') as f:
    raw_data = [row.strip().split('\t', 1) for row in f.readlines()]

is_spam = np.array([True if row[0] == 'spam' else False for row in raw_data])
messages = np.array([row[1].replace('&lt;', '<') \
                           .replace('&gt;', '>') \
                           .replace('&amp;', '&') \
                           .replace('\x92', "'")
     for row in raw_data], dtype=object)

df = pd.DataFrame({'message': messages, 'is_spam': is_spam})
print(df.describe(include='all'))


"""
Our dataset contains a total of 5574 messages, with 4827 hams (86.6%) and 747 spams (13.4%).
Some messages are duplicates. We choose not to remove them, they certainly
represent frequent messages for which making a classification error should be penalized more.
"""

# %% EDA

"""
## <a id="eda"></a> Preliminary data analysis

The messages are quite complex, mixing  uppercase, lowercase, puctuation, alphanumeric and other exotic characters.
Training a model to the data would certainly require some form of text normalization. However, this might cause a loss
of information. In this section we try to somewhat keep this information and aggregate it in the form of a few descriptive
to get insight about what makes a message a spam.

An important remark is in order before proceeding with data analysis. Visual inspection of the dataset reveals
the presence of some tokens specific to ham messages: `<#>`, `<DECIMAL>`, `<TIME>` and `<URL>`
(they appear as `&lt;*&gt;` in the dataset).
These were certainly introduced to mask personal information from hams. Although the kind of values that
`<DECIMAL>`, `<TIME>` and `<URL>` replace is obvious, this is not the case for `<#>`, which seems to be used
in replacement of any kind of value.
The character `*` seems to also be used to replace personal information in some ham messages
(see for instance message 983). However, this character is also present as such in messages.
"""

tokens_df = pd.DataFrame(
    {'is_spam': df['is_spam'],
     'count_<#>': df['message'].map(lambda x: x.count('<#>')),
     'count_<DECIMAL>': df['message'].map(lambda x: x.count('<DECIMAL>')),
     'count_<TIME>': df['message'].map(lambda x: x.count('<TIME>')),
     'count_<URL>': df['message'].map(lambda x: x.count('<URL>')),
     'count_*': df['message'].map(lambda x: x.count('*'))})

tokens_df.groupby('is_spam').sum()

"""
Indeed, these tokens are specific to hams (except the character `*`).
There presence, due to human annotations, could introduce
important biases to an automated detection system if not handled properly.

With this remark in mind, we can move to the characterization of messages using the following descriptive features:
- `msg_len`, the length of the message;
- `chr_caps`, the number of capitallized characters;
- `chr_digit`, the number of digit characters in the message;
- `chr_punct`, the number of punctuation characters in the message;
- `max_wd_len`, he length of the longest character string (split by whitespaces);
- `lex_money`, the number of some money-related words and characters: `'$'`, `'£'`, `'€'`, `'cash'`, `'free'`, `'price'` and `'prize'`.
The presence of these tokens indicates that the message talks about money (after all, spamming is about getting money from people).
We could also consider adding words related to the lexical field of sex.
- `caps_first`, the number of consecutive capitallized characters at the beginning of the message.

For the purpose of data visualization, we remove the duplicates in the dataset.
"""

## Define some helper functions
def count_tokens(msgs: pd.DataFrame, tokens: list['str'])-> pd.DataFrame:
    """
    Count the number of tokens in the token list from the casefolded messages.
    """
    regex = '|'.join(tokens)
    return msgs.apply(lambda x: len(re.findall(regex, x.casefold())))
    

def count_first_caps(s: str)-> int:
    """
    Count the number of capitallized letters at the beginning of a string.
    """
    n = 0
    for c in s:
        if not c.isupper():
            return n
        n += 1
    return n


def build_features(msgs: pd.DataFrame)-> pd.DataFrame:
    """
    Extract quantitative features from a series of messages.
    """
    data = {
        'msg_len': msgs.apply(len),
        'chr_caps': msgs.apply(lambda s: sum(c.isupper() for c in s)),
        'chr_digit': msgs.apply(lambda s: sum(c.isdigit() for c in s)),
        'chr_punct': msgs.apply(lambda s: sum(c in punctuation for c in s)),
        'max_wd_len': msgs.apply(lambda s: max(len(x) for x in s.split(' '))),
        'lex_money': count_tokens(msgs, [r'\$', '£', '€', 'free', 'cash', 'price', 'prize']),
        'caps_first': msgs.apply(count_first_caps)
        }
    return pd.DataFrame.from_dict(data)
 

## Prepare dataset from extracted features
df_ = df.drop_duplicates()
X = build_features(df_['message'])
y = df_['is_spam']


##
X_spam = X.loc[y]
X_spam.describe().map(lambda x: f"{x:.2f}")

##
X_ham = X.loc[~y]
X_ham.describe().map(lambda x: f"{x:.2f}")


"""
Some of our derived features are actually quite discriminating for the spam/ham nature of the messages.
- The number of digits is clearly the most discriminating feature with a mean of 15 for spams vs 0.3 for hams.
75 % of spams have more than 10 digits while less than 25% have a single digit.
- The presence of money-related words 'free', 'cash' and currency characters is also discriminative of spams, although many
spams do not have any.
- Spams tend to also have larger message lengths, capitallized characters and punctuation characters. However, there is more
overlap between the two categories in this case.
"""


v1, v2 = 'chr_digit', 'max_wd_len'

## number of hams with given values of `v1` and `v2`
count_ham = X_ham.loc[:, [v1, v2]].value_counts()
x_ham, y_ham = count_ham.index.to_frame().to_numpy().T

## number of spams with given values of `v1` and `v2`
count_spam = X_spam.loc[:, [v1, v2]].value_counts()
x_spam, y_spam = count_spam.index.to_frame().to_numpy().T


##
fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, figsize=(5.5, 4.8), dpi=200,
    gridspec_kw={'left': 0.12, 'right': 0.92, 'top': 0.9, 'bottom': 0.10}
)
fig1.suptitle('Figure 1: Scatter plot of messages characteristics',
              x=0.02, ha='left')

sc_ham = ax1.scatter(x_ham, y_ham, count_ham+2,
                     c='blue', alpha=0.65, marker='o', linewidths=0,
                     label='hams')
sc_spam = ax1.scatter(x_spam, y_spam, count_spam+2,
                      c='red', alpha=0.7, marker='o', linewidths=0,
                      label='spams')

ax1.set_xlim(-2, 50)
ax1.set_xticks(np.arange(5, 50, 5), minor=True)
ax1.set_xlabel('Number of digits `chr_digits`')

ax1.set_ylim(-0.5, 60)
ax1.set_ylabel('Max word length `max_wd_len`', labelpad=7)

ax1.grid(visible=True, which='major', linewidth=0.2)
ax1.grid(visible=True, which='minor', linewidth=0.1)

legend = ax1.legend()
legend.legend_handles[0]._sizes = [30]
legend.legend_handles[1]._sizes = [30]

plt.show()


"""
We show on figure 1 a scatter plot of the number of digits and max word length for both hams (blue) and spams (red).
The surface of the points is proportional to the number of messages (with an offset of 2 to facilitate vizualization).
Most hams have few digits and are made of short words. We expect that a simple linear model in our engineered feature space
will perform very well.
"""

# %% Utils
"""
!!! Clean the section
## <a id="utils"></a> Data preprocessing and utilities

Before moving on to model construction and training, we first introduce here some utilities related to model evaluation. We also setup here the common parts of the training pipelines.

### Preprocessing utilities


"""

def strip_diacritics(txt: str)-> str:
    """
    Remove all diacritics from words and characters forbidden in filenames.
    """
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)



"""
### Model evaluation utilities

We evaluate our classification models using metrics which are robust to the very high imbalance of fraud event occurences. These are derived from the confusion matrix:
- The precision
- The recall
- The F1 score
"""

def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray)-> None:
    """
    Helper function to evaluate and return the relevant evaluation metrics:
        confusion matrix, precision, recall, F1-score
    """
    cm = confusion_matrix(y_true, y_pred)
    
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1.
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1.
    f1 = 2*prec*recall/(prec+recall)
    
    return cm, prec, recall, f1



## Dataframe to hold the results
metric_names = ['precision', 'recall', 'F1-score']
index = pd.MultiIndex.from_product(
    [('Logistic regression', 'MLP - TFIDF', 'GRU - characters'),
     ('train', 'test')],
    names=['model', 'eval. set'])
evaluation_df = pd.DataFrame(
    np.full((6, 3), np.nan), index=index, columns=metric_names)






def plot_metrics(x: np.ndarray,
                 loss: np.ndarray, val_loss: np.ndarray,
                 prec: np.ndarray, val_prec: np.ndarray,
                 recall: np.ndarray, val_recall: np.ndarray,
                 f1_score: np.ndarray, val_f1_score: np.ndarray):
    """
    

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    loss, val_loss : np.ndarray
        DESCRIPTION.
    prec, val_prec : np.ndarray
        DESCRIPTION.
    recall, val_recall : np.ndarray
        DESCRIPTION.
    f1_score, val_f1_score : np.ndarray
        DESCRIPTION.

    """
    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(6.6, 5.8), dpi=200, sharex=True,
        gridspec_kw={'left': 0.08, 'right': 0.975, 'top': 0.92, 'bottom': 0.08,
                     'wspace': 0.18, 'hspace': 0.18}
    )
    
    for ax in axs.ravel():
        ax.grid(visible=True, linewidth=0.3)
        ax.set_xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x), len(x)//5))

    ## loss
    l1, = axs[0, 0].plot(x, loss, color='tab:blue')
    l2, = axs[0, 0].plot(x, val_loss, color='tab:orange')
    axs[0, 0].text(0.82, 0.9, 'Loss', fontsize=11,
                    transform=axs[0, 0].transAxes,
                    bbox={'boxstyle': 'round,pad=0.4', 'facecolor': '0.96'})

    ## precision
    axs[0, 1].plot(x, prec, color='tab:blue')
    axs[0, 1].plot(x, val_prec, color='tab:orange')
    axs[0, 1].set_ylim(0.65, 1)
    axs[0, 1].text(0.62, 0.1, 'Precision', fontsize=11,
                   transform=axs[0, 1].transAxes,
                   bbox={'boxstyle': 'round,pad=0.4', 'facecolor': '0.96'})

    ## recall
    axs[1, 0].plot(x, recall, color='tab:blue')
    axs[1, 0].plot(x, val_recall, color='tab:orange')
    axs[1, 0].set_ylim(0.65, 1)
    axs[1, 0].text(0.62, 0.1, 'Recall', fontsize=11,
                   transform=axs[1, 0].transAxes,
                   bbox={'boxstyle': 'round,pad=0.4', 'facecolor': '0.96'})

    ## F1-score
    axs[1, 1].plot(x, recall, color='tab:blue')
    axs[1, 1].plot(x, val_recall, color='tab:orange')
    axs[1, 1].set_ylim(0.65, 1)
    axs[1, 1].text(0.69, 0.08, 'F1-score', fontsize=11,
                   transform=axs[1, 1].transAxes,
                   bbox={'boxstyle': 'round,pad=0.4', 'facecolor': '0.96'})

    fig.text(0.5, 0.015, 'Epoch #', fontsize=11, ha='center')
    fig.legend(handles=[l1, l2], labels=['train', 'validation'],
               loc=(0.63, 0.945), ncols=2)
    
    return fig, axs


# %% logreg
"""
## <a id="logreg"></a> Benchmarking with a logistic regression

We set up a simple logistic regression here as a benchmark for comparison with more advanced deep learning methods.


### Model construction and training

We keep things simple here: no parameter tuning or decision threshold adjustment.
We split the dataset into a train and test set, the latter containing 20% of the observations.
We will retain this split for the rest of the project.
"""

## Simple train-test split, use the dataset with duplicates
X_tr, X_test, y_tr, y_test = train_test_split(
    build_features(df['message']), df['is_spam'],
    test_size=0.2, stratify=df['is_spam'], random_state=SEED)

## pipeline
base_lr = LogisticRegression(class_weight='balanced')
pipeline = Pipeline([('column_preprocessing', StandardScaler()),
                     ('classifier', base_lr)])

## train
t0 = time.time()
lr_model = pipeline.fit(X_tr, y_tr)
t1 = time.time()
print(f'Logistic regression model training in {t1-t0:.2f} s')


"""
### Interpretation
"""

feature_names = lr_model['column_preprocessing'].get_feature_names_out()
coefs = lr_model['classifier'].coef_[0]
for fn, c in zip(feature_names, coefs):
    print(f'{fn:<10} : {c: .4f}')


"""
The most determinant feature is clearly the number of digits in the message (spams have more),
followed by the number of punctuation characters (spams have less).
The other features are also good predictors, with the exception of the number of capitallized letters and their number
at the beginning of the message.
"""


#%%
"""
### Model evaluation

When considering the performance of our model, we must keep in mind that the masking of some
spam words introduce biases. This is especially the case for the masking of digits, the most important
feature for the classifiation. The evaluation below thus certainly overestimates the performance. 
"""

## evaluate of train set
y_pred_tr = lr_model.predict(X_tr)
_, prec, recall, f1 = classification_metrics(y_tr, y_pred_tr)
evaluation_df.iloc[0] = prec, recall, f1

## evaluate on test set
y_pred_test = lr_model.predict(X_test)
_, prec, recall, f1 = classification_metrics(y_test, y_pred_test)
evaluation_df.iloc[1] = prec, recall, f1


print(evaluation_df)

"""
Not bad! We get a benchmark F1-score of about 0.9. Let us see if we can improve this score.
"""
# %% TF-IDF MLP
"""
## <a id="tfidf-mlp"></a> A first neural network: Multi-layer perceptron and TF-IDF

!!!

### Text preprocessing

The text prepocessing proceeds in 4 steps:
1. Remove diacritics and casefold the messages.
2. Lemmatize words. However, we keep stop words as valid tokens.
3. Replace tokens containing digits with a special token `'<num>'`. This includes the tokens `<DECIMAL>` and `<TIME>` from hams.
Also discard the ham-specific token `<#>`.
4. Finally, replace tokens with a single occurrence (including the ham-specific token `<URL>`) with an out-of-vocabulary token (`<oov>`).
This processing is currently not available with scikit-learn `CountVectorizer`, so we do it manually.

Most of the messages are written in [SMS language](https://en.wikipedia.org/wiki/SMS_language) with inconsistent style.
Our tokenization will produce different tokens for words that have essentially the same meaning.
"""


## step 1: remove diacritics and preprocess
norm_msgs = df['message'].apply(strip_diacritics).apply(str.casefold)
norm_msgs.head()


# %%

## step 2: lemmatize words, remove punctuation
nlp = spacy.load('en_core_web_sm')
corpus = []
for msg in norm_msgs:
    doc = nlp(msg)
    doc = ' '.join(wd.lemma_ for wd in doc if not wd.is_punct)
                   # if wd.is_alpha and not wd.is_stop)
    corpus.append(doc)
corpus = np.array(corpus, dtype=object)


# %%
## step 3: replace numbers and specific tokens
preprocessed_corpus = np.empty_like(corpus, dtype=object)
for i, msg in enumerate(corpus):
    msg = msg.replace(' < > ', '') # discard `< # >` token
    msg = re.sub(r'< decimal >|< time >', '<num>', msg) # replace number masking tokens with `<num>`
    msg = re.sub(r'\b[,\.\-\w]*\d[,\.\-\w]*\b', '<num>', msg) # replace numbers with `<num>`
    msg = re.sub(r'\s+', ' ', msg) # replace consecutive whitespaces with single whitespace
    preprocessed_corpus[i] = msg


"""
!!! This gives ...
"""

##
print(corpus[12])
print('----------')
print(preprocessed_corpus[12])

##
print(corpus[802])
print('----------')
print(preprocessed_corpus[802])

##
print(corpus[2408])
print('----------')
print(preprocessed_corpus[2408])


# %%

## Step 4: manual handling of singwords
text = ' ## '.join(preprocessed_corpus)

vocab, counts = np.unique(text.split(' '), return_counts=True)
single_words = vocab[counts == 1]
for wd in single_words:
    text = text.replace(f' {wd} ', ' <oov> ')
processed_corpus = np.array(text.split(' ## '), dtype=np.str_)


# %%
"""
### Model construction and training
"""

## Construct the TF-IDF matrix
vectorizer = CountVectorizer(token_pattern=r'[^\s]+')
word_counts = vectorizer.fit_transform(processed_corpus)
tfidf = TfidfTransformer(norm='l2').fit_transform(word_counts)
tfidf

## Train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    tfidf, df['is_spam'], test_size=0.2, stratify=df['is_spam'],
    random_state=SEED)

## MLP classifer, no preprocessing pipeline
mlp_model = MLPClassifier(hidden_layer_sizes=(1000, 300, 100, 10),
                          solver='adam', alpha=1e-1,
                          max_iter=300, random_state=SEED)

t0 = time.time()
mlp_model.fit(X_tr, y_tr)
t1 = time.time()
print(f'Multi-layer perceptron training in {t1-t0:.2f} s')


# %%
"""
### Model evaluation
"""

## evaluate of train set
y_pred_tr = mlp_model.predict(X_tr)
_, prec, recall, f1 = classification_metrics(y_tr, y_pred_tr)
evaluation_df.iloc[1] = prec, recall, f1

## evaluate on test set
y_pred_test = mlp_model.predict(X_test)
_, prec, recall, f1 = classification_metrics(y_test, y_pred_test)
evaluation_df.iloc[3] = prec, recall, f1


print(evaluation_df)

"""
The F1-score reaches a value of 93.1% on the test set, a significative improvement
over the simple linear regression. However, this comes at the cost of a much longer
training time. Another drawback is the tendency of the model to overfit, as
indicated by the difference between train and test F1-scores.
"""

# %% GRU chars
"""
## <a id="gru-char"></a> Gated Recurrent Units with character tokens

The messages are rather short, we can therefore consider a tokenization at the character level.
"""

##
df['message'].apply(len).describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99, 0.999])

"""
The messages have a maximal length of 910 characters, with less than 50 messages
longer than 300 characters.
"""

"""
### Text preprocessing

Handling the messages at the character level imposes its own processing. In
particular, we must manage the ham-specific tokens.
- The numeric tokens `<DECIMAL>` and `<TIME>` are replaced by a random digit.
- The tokens `<#>` and `<URL>` are simply discarded.
- The characters with less than 10 counts are replaced by an out-of-vocabulary token
- The message sequence length is truncated to 200 characters. This operation affects 109
messages (2%) of the dataset.
"""

## Setup the corpus
gru_corpus = np.array([m.replace('<DECIMAL>', f'{rng.integers(0, 10)}') \
                        .replace('<TIME>', f'{rng.integers(0, 10)}') \
                        .replace('<#>', '').replace('<URL>', '')
                       for m in df['message']])

## create vocabluary keep chars that occur more than 10 times
text_chars = np.array([''.join(gru_corpus)]).view('<U1')
chars, counts = np.unique(text_chars, return_counts=True)
chars[np.argsort(counts)], np.sort(counts)

## setup text vectorizer
vocabulary = chars[counts >= 10]
chars_encoder = keras.layers.TextVectorization(
    standardize=None, split='character', vocabulary=vocabulary,
    output_sequence_length=200)


# %%
"""
### Dataset construction

"""

## train - validation - test split of the data
is_spam = df['is_spam'].to_numpy()
X_, X_test, y_, y_test = train_test_split(
    gru_corpus, is_spam, test_size=0.2, stratify=is_spam, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_, y_, test_size=0.2, stratify=y_, random_state=SEED)


## construction of tensorflow Datasets
batch_size = 32
ds_tr = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(batch_size)
ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)


# %%
"""
### Model construction

!!!
- masking to handle messages of variable length
- Initialize bias
We set inital bias to speedup training. See
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=en
https://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
"""
## Compute bias 
p_spam = df['is_spam'].mean()
bias_init = keras.initializers.Constant(np.log(p_spam / (1 - p_spam)))
p_spam

## Setup early stopping
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

## GRU model
gru_chars_model = keras.models.Sequential(
    [chars_encoder,
     keras.layers.Embedding(
         input_dim=len(chars_encoder.get_vocabulary()),
         output_dim=64, mask_zero=True),
     keras.layers.GRU(units=64, return_sequences=True),
     keras.layers.GRU(units=32),
     keras.layers.Dense(16, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias_init)],
    name='GRU_chars_model',
)

gru_chars_model.compile(optimizer='nadam',
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[keras.metrics.Precision(name='precision'),
                                 keras.metrics.Recall(name='recall'),
                                 # keras.metrics.F1Score(name='F1-score')
                                 ])

# %%
##

t0 = time.time()
history = gru_chars_model.fit(ds_tr, epochs=50, validation_data=ds_val,
                              verbose=2, callbacks=[early_stop_cb])
t1 = time.time()
print(f'Gated recurrent unit model training in {t1-t0:.2f} s')

# %%
"""
### Model evaluation

!!!
"""

x = history.epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
prec = np.array(history.history['precision'])
val_prec = np.array(history.history['val_precision'])
recall = np.array(history.history['recall'])
val_recall = np.array(history.history['val_recall'])
f1_score = 2 * prec*recall / (prec + recall)
val_f1_score = 2 * val_prec*val_recall / (val_prec + val_recall)


fig2, axs2 = plot_metrics(history.epoch, loss, val_loss,
                          prec, val_prec, recall, val_recall,
                          f1_score, val_f1_score)

fig2.suptitle('Figure 2: Evaluation metrics', x=0.02, ha='left')
plt.show()



# %%
##
print('===== Train set metrics =====')
y_pred_tr = (gru_chars_model.predict(X_tr).ravel() > 0.5)
print_metrics(confusion_matrix(y_tr, y_pred_tr))

##
print('===== Validation set metrics =====')
y_pred_val = (gru_chars_model.predict(X_val).ravel() > 0.5)
print_metrics(confusion_matrix(y_val, y_pred_val))

##
print('\n===== Test set metrics =====')
y_pred_test = (gru_chars_model.predict(X_test).ravel() > 0.5)
print_metrics(confusion_matrix(y_test, y_pred_test))


"""
!!!
"""


# %% GRU words
"""
## <a id="gru-word"></a> Gated Recurrent Units with word tokens



### Text preprocessing

The text prepocessing is similar to that of the TF-IDF encoding:
1. Remove diacritics and casefold the messages.
2. Lemmatize words, keep stop words as valid tokens.
3. Replace tokens containing digits with a special token `'<num>'`. This includes the tokens `<DECIMAL>` and `<TIME>` from hams.
Also discard the ham-specific token `<#>`.
4. Finally, replace tokens with a single occurrence (including the ham-specific token `<URL>`) with an out-of-vocabulary token (`<oov>`).
This processing is currently not available with scikit-learn `CountVectorizer`, so we do it manually.

"""


chars_encoder = keras.layers.TextVectorization(
    standardize=None, split='character', vocabulary=vocabulary,
    output_sequence_length=200)
chars_ohe = keras.layers.CategoryEncoding(
    chars_encoder.vocabulary_size(), output_mode='one_hot')

chars_ohe = keras.layers.IntegerLookup(
     output_mode='one_hot',
    vocabulary=np.arange(chars_encoder.vocabulary_size()))



## GRU model
gru_words_model = keras.models.Sequential(
    [chars_encoder,
     keras.layers.Embedding(
         input_dim=len(chars_encoder.get_vocabulary()),
         output_dim=64, mask_zero=True),
     keras.layers.GRU(units=64, return_sequences=True),
     keras.layers.GRU(units=32),
     keras.layers.Dense(16, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias_init)],
    name='GRU_words_model',
)

gru_words_model.compile(optimizer='nadam',
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[keras.metrics.Precision(),
                                 keras.metrics.Recall(),
                                 # keras.metrics.F1Score()
                                 ])
# %%
##
history = gru_words_model.fit(ds_tr, epochs=1, validation_data=ds_val,
                              verbose=2)


# %%
"""
### Model evaluation


"""





##
print('===== Train set metrics =====')
print_metrics(confusion_matrix(y_tr, gru_words_model.predict(X_tr)))

##
print('===== Validation set metrics =====')
print_metrics(confusion_matrix(y_val, gru_words_model.predict(X_val)))

##
print('\n===== Test set metrics =====')
print_metrics(confusion_matrix(y_test, gru_words_model.predict(X_test)))






















# %%

# preprocessed_corpus = np.load('preprocessed_corpus.npy', allow_pickle=False)
unique_tokens, token_counts = np.unique(sum(norm_msgs, start=[]), return_counts=True)
unique_tokens = unique_tokens[np.argsort(token_counts)][::-1] # decreasing order
token_counts = np.sort(token_counts)[::-1]

np.sum(token_counts)

##
vocab_size = 500

print(f'Vocabulary totaling {np.sum(token_counts[:vocab_size])} counts\n')
k = 10
print('========== Top tokens (rank, token, count) ==========')
for i in range(k):
    print('||   ', end='')
    for j in range(3):
        n = i + k*j
        print(f'({n+1:>2})   {unique_tokens[n]:<7} : {token_counts[n]:>4}',
              end='   ||   ')
    print()

print('\n========== Last retained tokens (rank, token, count) ==========')
for i in range(k, 0, -1):
    print('||   ', end='')
    for j in range(3, 0, -1):
        n = vocab_size-i-k*j + k
        print(f'({n+1})   {unique_tokens[n]:<14} : {token_counts[n]:>4}',
              end='   ||   ')
    print()








sys.exit()

# %% text preprocessing

"""
## <a id="text_preprocessing"></a> Text preprocessing

Most of the messages are written in [SMS language](https://en.wikipedia.org/wiki/SMS_language) with inconsistent style.
A blind tokenization of the words would produce different tokens for words that have essentially the same meaning.


Our text prepocessing proceeds in 5 steps:
1. Remove diacritics and casefold the messages. Some characters appear to have diacritics, possibly a text encoding issue.
2. Discard characters with low count in the corpus
3. Lemmatize, remove stop words and spaces. We keep punctuation since it was shown to be an important discrimination factor.
4. Replace tokens containing alphanumeric characters with a special token `'<num>'`. These are likely to be discarded in any encoding with reasonable
vocabulary size. This is bad since it is a highly relevant factor for spam discrimination.


#### Step 1
"""

# TODO add columns
## step 1 remove diacritics and preprocess
def strip_diacritics(txt: str)-> str:
    """
    Remove all diacritics from words and characters forbidden in filenames.
    """
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)

norm_msgs = df['message'].apply(strip_diacritics).apply(str.casefold)
norm_msgs.head()





#%%

"""
#### Step 2
"""

def remove_chars(text: str, chars: str)-> str:
    """
    Replace the selected characters by whitespaces in the text.
    """
    for c in chars:
        if c in text:
            text = text.replace(c, ' ')
    return text

##
char_counts = Counter(''.join(norm_msgs))
char_counts

##
chars = ''.join(c for c, v in char_counts.items() if v < 20)
chars += '():' # 

##
norm_msgs = norm_msgs.apply(remove_chars, chars=chars)
norm_msgs.head()


"""
#### Step 3
"""
# download with `python -m spacy download en_core_web_sm`
nlp = spacy.load('en_core_web_sm')
def lemmatize(text: str)-> list[str]:
    """
    Split a text into tokens.
    The tokens are lemmatized, spaces and stop words are removed.
    """
    doc = nlp(text)
    return [wd.lemma_ for wd in doc if not (wd.is_stop or wd.is_space)]

norm_msgs = norm_msgs.apply(lemmatize)
norm_msgs.head()


"""
#### Step 4

"""

def substitute_num(tokens: list[str], subs: str = '<num>')-> list[str]:
    """
    Substitute tokens containing digits by a special token `subs`.
    """
    return [subs if any(c.isdigit() for c in t) else t for t in tokens]

norm_msgs = norm_msgs.apply(substitute_num)
norm_msgs = norm_msgs.apply(lemmatize)
norm_msgs.head()

data = norm_msgs.to_numpy()


# %% 
"""
## <a id="model1"></a> A first model


#### Step 1
"""

# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_text as tf_text

"""
### Tokenize and pad

We retain a vocabulary size of 500 tokens, the less frequent having only 16 counts accross the whole corpus.

, the most importants are the punctuation
"""
##
unique_tokens, token_counts = np.unique(sum(norm_msgs, start=[]), return_counts=True)
unique_tokens = unique_tokens[np.argsort(token_counts)][::-1] # decreasing order
token_counts = np.sort(token_counts)[::-1]

np.sum(token_counts)

##
vocab_size = 500

print(f'Vocabulary totaling {np.sum(token_counts[:vocab_size])} counts\n')
k = 10
print('========== Top tokens (rank, token, count) ==========')
for i in range(k):
    print('||   ', end='')
    for j in range(3):
        n = i + k*j
        print(f'({n+1:>2})   {unique_tokens[n]:<7} : {token_counts[n]:>4}',
              end='   ||   ')
    print()

print('\n========== Last retained tokens (rank, token, count) ==========')
for i in range(k, 0, -1):
    print('||   ', end='')
    for j in range(3, 0, -1):
        n = vocab_size-i-k*j + k
        print(f'({n+1})   {unique_tokens[n]:<14} : {token_counts[n]:>4}',
              end='   ||   ')
    print()


sys.exit()

# %%

import tensorflow as tf
from tensorflow import keras
# import tensorflow_text as tf_text

## tokenize
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                               char_level=False,
                                               oov_token='out_of_vocab')
tokenizer.fit_on_texts(data)
data = tokenizer.texts_to_sequences(data)

## build dataset
seed = 1234
ds = tf.data.Dataset.from_tensor_slices((data, df['is_spam'].to_numpy()))
ds_tr, ds_val = keras.utils.split_dataset(ds, left_size=0.8, shuffle=True, seed=seed)
ds_tr = ds_tr.shuffle(buffer_size=10, seed=seed).batch(64)
ds_val = ds_val.shuffle(buffer_size=10, seed=seed).batch(64)


"""
We set inital bias to speedup training. See
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=en
https://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
"""
## get initial bias 
p_spam = df['is_spam'].mean()
bias = np.log(p_spam / (1 - p_spam))
p_spam

## make model with initialized bias
model = keras.models.Sequential(
    [keras.layers.Embedding(vocab_size, 32, name='embedding'),
     keras.layers.GRU(units=64, return_sequences=True),
     keras.layers.GRU(units=32),
     keras.layers.Dense(16, activation='relu'),
     keras.layers.Dense(1, activation='sigmoid', bias_initializer=bias)],
    name='RNN1',
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(ds_tr, epochs=50, validation_data=ds_val)



# %%
"""
ChatGPT says use fasttext

Do:
-> basic cleaning, lematization and bag of words
   -> simple MLP model
-> embedding with fasttext then RNR / transformer / MLP ?
"""



# %% Conclusion
"""
## <a id="conclusion"></a> Conclusion and perspectives


"""

