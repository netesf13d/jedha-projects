# -*- coding: utf-8 -*-
"""
Script version of the North Face ecommerce project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import wordcloud


# %% Loading and preprocessing
"""
## <a name="loading"></a> Data loading and preprocessing
"""

df = pd.read_csv('../sample-data.csv')

descriptions = df['description'].to_numpy()


# download with `python -m spacy download en_core_web_sm`
nlp = spacy.load('en_core_web_sm')
def lemmatize(text: str)-> list[str]:
    """
    Split a text into tokens.
    The tokens are lemmatized, spaces and stop words are removed.
    """
    doc = nlp(text)
    return [wd.lemma_ for wd in doc if not (wd.is_stop or wd.is_space)]

# norm_msgs = norm_msgs.apply(lemmatize)
# norm_msgs.head()



# %% clustering
"""
## <a id="clustering"></a> Descriptions clustering

"""



# %% recommender system
"""
## <a id="recommendation"></a> Recommender system

"""


# %% topic modeling
"""
## <a id="topic"></a> Topic modeling

"""

