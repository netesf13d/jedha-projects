# -*- coding: utf-8 -*-
"""
Script for the AT&T spam detector project
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


# %% Loading
"""
## <a name="loading"></a> Data loading and preprocessing
"""

# Load data
df = pd.read_csv('./spam.csv', encoding='mac-roman')

##
df.describe()

"""
The dataset consists of 5572 observations, and has 5 columns. Of these columns, only the first two
are named while the others are not.
"""

df.loc[~df.iloc[:, 4].isna()]

"""
Visual inspection on notepad++ shows that the unamed columns and the trailing comas are there because
they act both as a delimiter of the CSV file and as a character in some messages.
This is highlighted in the following screen captures.
"""

# replace NaNs with the empty string
df = df.replace(float('nan'), '')

# concatenate columns >= 1 to get the messages
# convert target to a boolean series
df = pd.DataFrame.from_dict({'message': df.iloc[:, 1:].apply(''.join, axis=1),
                             'is_spam': df.iloc[:, 0] == 'spam'})

"""
Some of the files entries have enclosing quotes (eg the first one, see the picture above), while others do not (eg, the second one).
Our preprocessing removed them. It is not clear whether those are relevant to the dataset, but in the remaining we will assume they are not.
"""

##
df.describe()

"""
Our dataset seems to have duplicates. We choose not to remove them, they certainly
represent frequent messages for which making a classification error should be penalized more.
"""

# %% EDA

"""
## <a name="eda"></a> Preliminary data analysis

The messages are quite complex, mixing  uppercase, lowercase, puctuation, alphanumeric and other exotic characters.
Training a model to the data would certainly require some form of text normalization. However, this might cause a loss
of information.

To get some insights about what makes a message a spam and make visualizations, we will reduce the messages to a few descriptive quantities:
- The length of the message, `msg_len`;
- The number of capitallized characters, `chr_caps`;
- The number of digit characters in the message, `chr_digit`;
- The number of punctuation characters in the message, `chr_punct`;
- The length of the longest character string (split by whitespaces), `max_wd_len`;
- The number of some money-related words and characters `'$'`, `'£'`, `'€'`, `'cash'`, `'free'`, `'price'` and `'prize'`, `lex_money`.
The presence of these tokens indicates that the message semantics lies in the lexical field of money
(after all, spamming is about getting money from people). We could also consider adding words related to the lexical field of sex.
# - The number of (casefolded) `'cash'` strings, `wd_cash`;
# - The number of (casefolded) `'free'` strings, `wd_free`;

For the purpose of data visualization, we remove the duplicates in the dataset.
"""

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
        'lex_money': count_tokens(msgs, ['\$', '£', '€', 'free', 'cash', 'price', 'prize']),
        'caps_first': msgs.apply(count_first_caps)
        # 'chr_curr': msgs.apply(lambda x: sum(c in '$£€' for c in x)),
        # 'wd_cash': msgs_cf.apply(lambda x: len(re.findall(r'cash', x))),
        # 'wd_free': msgs_cf.apply(lambda x: len(re.findall(r'free', x))),
        }
    return pd.DataFrame.from_dict(data)
 

df_ = df.drop_duplicates()
X = build_features(df_['message'])
y = df_['is_spam']


##
X_spam = X.loc[y]
X_spam.describe()

##
X_ham = X.loc[~y]
X_ham.describe()


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
count_ham = X_ham.loc[:, [v1, v2]].value_counts()
x_ham, y_ham = count_ham.index.to_frame().to_numpy().T

count_spam = X_spam.loc[:, [v1, v2]].value_counts()
x_spam, y_spam = count_spam.index.to_frame().to_numpy().T


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
ax1.set_xlabel('Number of digits')

ax1.set_ylim(-0.5, 60)
ax1.set_ylabel('Max word length', labelpad=7)

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

# %% logreg
"""
## <a name="logreg"></a> Benchmarking with a logistic regression

We set up a simple logistic regression here as a benchmark for comparison with more advanced deep learning methods.
"""

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def print_metrics(cm: np.ndarray,
                  print_cm: bool = True,
                  print_precrec: bool = True)-> None:
    """
    Print metrics related to the confusion matrix: precision, recall, F1-score.
    """
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1.
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1.
    if print_cm:
        print("Confusion matrix\n", cm / np.sum(cm))
    if print_precrec:
        print(f'Precision: {prec:.8}; recall: {recall:.8}')
    print(f'F1-score: {2*prec*recall/(prec+recall):.8}')


"""
### Model construction and training

We keep things simple here: no parameter tuning or decision threshold adjustment.
We split the dataset into a train and test set, the latter containing 20% of the observations.
# We will retain this split for the rest of the project.
"""

## Simple train-test split, use the dataset with duplicates
X_tr, X_test, y_tr, y_test = train_test_split(
    build_features(df['message']), df['is_spam'],
    test_size=0.2, stratify=df['is_spam'], random_state=1234)

# pipeline
pipeline = Pipeline([('column_preprocessing', StandardScaler()),
                     ('classifier', LogisticRegression())])

# training
lr_model = pipeline.fit(X_tr, y_tr)

"""
### Interpretation
"""

feature_names = lr_model['column_preprocessing'].get_feature_names_out()
coefs = lr_model['classifier'].coef_[0]
for fn, c in zip(feature_names, coefs):
    print(f'{fn:<10} : {c: }')


"""
The most determinant feature is clearly the number of digits in the message (spams have more),
followed by the number of punctuation characters (spams have less).
The other features are also good predictors, with the exception of the number of capitallized letters and their number
at the beginning of the message.
"""


#%%
"""
### Model performance
"""

print('===== Train set metrics =====')
print_metrics(confusion_matrix(y_tr, lr_model.predict(X_tr)))

print('\n===== Test set metrics =====')
print_metrics(confusion_matrix(y_test, lr_model.predict(X_test)))

"""
Not bad! We get a benchmark F1-score of about 0.9. Let us see if we can improve this score.
"""

# %% test preprocessing

"""
## <a name="text_preprocessing"></a> Text preprocessing

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
## <a name="model1"></a> A first model


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





