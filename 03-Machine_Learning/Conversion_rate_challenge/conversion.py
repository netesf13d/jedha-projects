#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import time
# from itertools import product
# from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import Colormap, LightSource
# from matplotlib.collections import PolyCollection



# =============================================================================
# Data loading and initial preprocessing
# =============================================================================

## Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]


# test_df = pd.read_csv('./conversion_data_test.csv')
# test_df.describe(include='all')
# """
# The distribution of the different observations is essentially the same
# as that of the training set in terms of mean and standard deviation.
# """


#%%
# =============================================================================
# Preliminary EDA
# =============================================================================


def prob_distrib(df: pd.DataFrame,
                 group_by: str,
                 variables: list[str],
                 xvals: list[np.ndarray],
                 )-> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO doc

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    group_by : str
        DESCRIPTION.
    variables : list[str]
        DESCRIPTION.
    xvals : list[np.ndarray]
        DESCRIPTION.

    Returns
    -------
    group_vals : list
        DESCRIPTION.
    counts : np.ndarray
        DESCRIPTION.
    pconv : np.ndarray
        DESCRIPTION.
    std_pconv : TYPE
        DESCRIPTION.

    """
    group_vals = []
    counts = np.zeros((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                      dtype=float)
    pconv = np.full((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                    np.nan, dtype=float)

    for k, (group, gdf) in enumerate(df.groupby(group_by)):
        group_vals.append(group)
        for idx, c in gdf.loc[:, variables].value_counts().items():
            counts[k, *idx] = c
        pconv_df = gdf.loc[:, variables + ['converted']] \
                      .groupby(variables) \
                      .mean()['converted']
        for idx, p in pconv_df.items():
            idx = idx if isinstance(idx, tuple) else (idx,) # cast to tuple if no multiindex
            pconv[k, *idx] = p

    std_pconv = np.where(pconv * (1 - pconv) == 0., 1, pconv) / np.sqrt(counts)
    return group_vals, counts, pconv, std_pconv


def prob_distrib_plot(xvals: np.ndarray,
                      counts: np.ndarray,
                      pconv: np.ndarray,
                      std_pconv: np.ndarray,
                      group_labels: list[str]):
    """
    TODO doc

    Parameters
    ----------
    xvals : np.ndarray
        DESCRIPTION.
    counts : np.ndaarray
        DESCRIPTION.
    pconv : np.ndarray
        DESCRIPTION.
    std_pconv : np.ndarray
        DESCRIPTION.
    labels : list[str]
        DESCRIPTION.

    Returns
    -------
    fig : atplotlib Figure
        DESCRIPTION.
    axs : np.ndarray[matplotlib Axes]
        DESCRIPTION.

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    
    axs[0].grid(visible=True)
    distribs = counts / np.sum(counts, axis=1, keepdims=True)
    for k, label in enumerate(group_labels):
        axs[0].plot(xvals, distribs[k], color=f'C{k}',
                    linewidth=1, label=f'{label} ({int(np.sum(counts[k]))})')
    axs[0].set_xlim(0, np.max(xvals)+1)
    axs[0].set_ylim(-np.max(distribs)*0.02, np.max(distribs)*1.02)
    axs[0].set_ylabel('Prob. density')

    axs[1].grid(visible=True)
    for k, label in enumerate(group_labels):
        axs[1].errorbar(xvals, pconv[k], yerr=std_pconv[k],
                        color=f'C{k}', fmt='o', markersize=3, markeredgewidth=0.3,
                        label=label)
    axs[1].set_xlim(0, np.max(xvals)+1)
    axs[1].set_ylim(-0.005, int(np.max(pconv[~np.isnan(pconv)])*120)/100)
    axs[1].set_title("Conversion probability")
    axs[1].set_ylabel('Conversion probability')
    
    return fig, axs



#%% 

## 
df.groupby(['country', 'converted']).count()['age'] # the name 'age' is irrelevant here
##
df.groupby(['source', 'converted']).count()['age']

df1 = df.groupby(['new_user', 'source', 'country']).mean()['converted']
countries = ['China', 'Germany', 'UK', 'US']
sources = ['Ads', 'Direct', 'Seo']

# new user
new_pconv = np.empty((len(sources), len(countries)), dtype=float)
for i, source in enumerate(sources):
    for j, country in enumerate(countries):
        new_pconv[i, j] = df1.loc[1, source, country]

# recurring user
rec_pconv = np.empty((len(sources), len(countries)), dtype=float)
for i, source in enumerate(sources):
    for j, country in enumerate(countries):
        rec_pconv[i, j] = df1.loc[0, source, country]


# TODO finish plot
fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
axs1_cb = np.empty_like(axs1, dtype=object)

axs1[0].set_aspect('equal')
heatmap1 = axs1[0].pcolormesh(100*new_pconv, cmap='plasma',
                              edgecolors='k', lw=0.5)
axs1[0].set_xticks([0.5, 1.5, 2.5, 3.5], countries, rotation=60)
axs1[0].set_yticks([0.5, 1.5, 2.5], sources)
axs1[0].set_title("New visitors")
for (i, j), p in np.ndenumerate(new_pconv):
    color = 'k' if p > np.max(new_pconv) / 3 else 'w'
    axs1[0].text(j+0.5, i+0.5, f'{100*p:.2f}', color=color,
                 ha='center', va='center')

divider1 = make_axes_locatable(axs1[0])
axs1_cb[0] = divider1.append_axes("right", size="7%", pad="5%")
fig1.add_axes(axs1_cb[0])
fig1.colorbar(heatmap1, cax=axs1_cb[0], orientation="vertical",
              ticklocation="right")
axs1_cb[0].set_yticks([0, 0.5, 1, 1.5, 2, 2.5])


axs1[1].set_aspect('equal')
heatmap2 = axs1[1].pcolormesh(100*rec_pconv, cmap='plasma',
                              edgecolors='k', lw=0.5)
axs1[1].set_xticks([0.5, 1.5, 2.5, 3.5], countries, rotation=60)
axs1[1].tick_params(left=False, labelleft=False)
axs1[1].set_title("Recurring visitors")
for (i, j), p in np.ndenumerate(rec_pconv):
    color = 'k' if p > np.max(rec_pconv) / 3 else 'w'
    axs1[1].text(j+0.5, i+0.5, f'{100*p:.2f}', color=color,
                 ha='center', va='center')

divider2 = make_axes_locatable(axs1[1])
axs1_cb[1] = divider2.append_axes("right", size="7%", pad="5%")
fig1.add_axes(axs1_cb[1])
fig1.colorbar(heatmap2, cax=axs1_cb[1], orientation="vertical",
              ticklocation="right")
axs1_cb[1].set_yticks([0, 5, 10, 15])


plt.show()

"""
Here we show heatmaps of the newsletter subscription probabilility for the
categorical features `'source'` and `'country'`, for both new user and recurring users.
We can make the following remarks:
- Recurring visitors are 2.5 - 8 times more likely to subscribe to the newsletter.
- Chinese visitors are much less likely to subscribe than others, and US visitors tend to subscribe less than europeans.
- The way by which visitors reached the website has little impact on their eventual subscription.
"""


#%%
### Age data
## prepare age data
ages = np.arange(80)
new_user, age_counts, age_pconv, age_std_pconv = prob_distrib(
    df, group_by='new_user', variables=['age'], xvals=[ages])


## let us make a nice exponential fit
from scipy.optimize import curve_fit

def decay_exp(x: np.ndarray,
              A: float,
              tau: float)-> np.ndarray:
    """
    Exponential decay: f(x) = A * exp(- x / tau)
    """
    return A * np.exp(- x / tau)

idx = ~np.isnan(age_pconv)
popt, pcov = [], []
for k, i in enumerate(idx):
    popt_, pcov_ = curve_fit(decay_exp, ages[i], age_pconv[k, i], p0=(0.1, 10),
                             sigma=age_std_pconv[k, i])
    popt.append(popt_)
    pcov.append(pcov_)


# ## plot
fig2, axs2 = prob_distrib_plot(
    ages, age_counts, age_pconv, age_std_pconv,
    group_labels=['recur.', 'new'])

axs2[0].text(40, 0.035, f'total counts: {int(np.sum(age_counts))}')
axs2[0].axvline(17, color='k', linewidth=1, linestyle='--')
axs2[0].text(11, 0.04, '17 yo\ncutoff', ha='center', va='center')
axs2[0].set_title("Age distribution of visitors")
axs2[0].set_xlabel('Age (years)')
axs2[0].legend(loc=1)

for k, popt_ in enumerate(popt):
    axs2[1].plot(ages, decay_exp(ages, *popt_),
                 linewidth=1, color=f'C{k}', label='exponential fit')
popt_txt = [f'A = {popt_[0]:.3} +- {pcov_[0, 0]:.2}\ntau = {popt_[1]:.3} +- {pcov_[1, 1]:.2}'
            for popt_, pcov_ in zip(popt, pcov)]
axs2[1].text(35, 0.12, popt_txt[0], ha='center', va='center', fontsize=7)
axs2[1].text(14, 0.05, popt_txt[1], ha='center', va='center', fontsize=7)
axs2[1].axvline(17, color='k', linewidth=1, linestyle='--')
axs2[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], [0, 5, 10, 15, 20])
axs2[1].set_ylabel('Conversion probability (%)')
axs2[1].set_title("Conversion probability vs age")
axs2[1].set_xlabel('Age (years)')
axs2[1].legend()

plt.show()

"""
Here we show the distribution of ages (left panel) and the conversion probability (right panel)
aggregated by `'source'` and `'country'`, but distinguishing new and recurrent visitors.
- The distribution of ages is the same for the two categories: a truncated gaussian with a cutoff at 17 years old,
  centered at around 28 years old. The probability of having a new visitor is roughly twice that of a recurrent visitor.
- The newsletter subscription probability decays with age. The observed decay fits well with an exponential
  from which we recover very similar decay constants of about 14 years. The multiplicative factor is 5-6 times higher
  for recurrent visitors than for new visitors, and so is the subscription probability both both categories, almost *independently of age*.

These patterns are extremely regular and again reveal the artificial nature of the data. They would certainly not be occur in a realistic context.
"""


#%% 

### Number of pages visited
"""
We proceed in a similar fashion this time with the quantity `'total_pages_visited'`
"""
npages = np.arange(30)
new_user, npage_counts, npage_pconv, npage_std_pconv = prob_distrib(
    df, group_by='new_user', variables=['total_pages_visited'], xvals=[npages])


## Your beloved sigmoid fit
def sigmoid(x: np.ndarray,
            x0: float,
            a: float)-> np.ndarray:
    """
    Exponential decay: f(x) = 1 / (1 + exp(- a * (x - x0)))
    """
    return 1 / (1 + np.exp(- a * (x - x0)))

idx = ~np.isnan(npage_pconv)
popt, pcov = [], []
for k, i in enumerate(idx):
    popt_, pcov_ = curve_fit(sigmoid, npages[i], npage_pconv[k, i], p0=(15, 1))
    popt.append(popt_)
    pcov.append(pcov_)


## plot
fig3, axs3 = prob_distrib_plot(
    npages, npage_counts, npage_pconv, npage_std_pconv,
    group_labels=['recur. visitors', 'new visitors'])

axs3[0].text(13, 0.123, f'total counts: {int(np.sum(npage_counts))}')
axs3[0].set_ylim(-0.005, 0.16)
axs3[0].set_title("# pages visited distribution")
axs3[0].set_xlabel('# pages visited')
axs3[0].legend()


for k, popt_ in enumerate(popt):
    axs3[1].plot(npages, sigmoid(npages, *popt_),
                 linewidth=1, color=f'C{k}', label='sigmoid fit')

popt_txt = [f'x0 = {popt_[0]:.3} +- {pcov_[0, 0]:.2}\na = {popt_[1]:.3} +- {pcov_[1, 1]:.2}'
            for popt_, pcov_ in zip(popt, pcov)]
axs3[1].text(7, 0.7, popt_txt[0], ha='center', va='center', fontsize=7)
axs3[1].text(21, 0.6, popt_txt[1], ha='center', va='center', fontsize=7)

axs3[1].set_xlim(0, 30)
axs3[1].set_ylim(-0.005, 1.005)
# axs3[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], [0, 5, 10, 15, 20])
# axs3[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2], minor=True)
axs3[1].set_title("Conversion probability")
axs3[1].set_xlabel('# pages visited')
axs3[1].set_ylabel('Conversion probability')
axs3[1].legend()

plt.show()


txt = """
Here we show the distribution of number of pages visited (left panel) and the corresponding conversion probability (right panel)
built the same way as in the previous figure.
- Although we still have a factor ~2 between the total counts, we remark that website visits with more than 13 pages visited are
  mostly done by recurrent visitors. Accounting for the factor 2, this makes it more than twice likely that a > 13 pages visit
  is done by a recurrent visitor.
- The two distributions are thus not proportional. They actually look like
  [inverse-gamma distributions](https://en.wikipedia.org/wiki/Inverse-gamma_distribution).
- The newsletter subscription probability looks like a sigmoid for both new and recurring visitors. The inflection points are shifted,
  and looking at the 13 pages-visits threshold, we see that here the conversion probability is thrice for recurring users than for new users.
- Sigmoid fits with the slope and inflection point as parameters is also shown on the right panel. The slopes are roughly the same,
  the difference being the inflection point: 12.5 pages for recurring visitors vs 15 pages for new visitors.
- Finally, we note two outlier events on the right panel. ~30 pages visits with no subscription. We might consider removing these observations
  for training later on.

We again stress that such regularity is unlikely in any realistic context.
"""

#%%

### Effect of the number of pages visited for each country

"""
We noted above that the country of origin has a significant impact on the conversion probability.
We consider the two following hypotheses to explain this trend:
- The country of orgin has an impact on the distribution of number of pages visited, but not on the subscription threshold (the inflection point of the sigmoid).
  In this case, the country is expected to not be a good a predictor: the relevant information would be already contained in the number of pages visited.
- The pages visits distribution does not depend on the country, but the conversion threshold does. In this case, the country (more precisely, its correlation with the number of pages visted)
  would be a highly relevant predictor.
For comparison, the new/recurring character of visitors effect is a mixture of these two: it affects both the pages visits distribution and conversion threshold.
"""

npages = np.arange(30)
country, npage_counts, npage_pconv, npage_std_pconv = prob_distrib(
    df, group_by='country', variables=['total_pages_visited'], xvals=[npages])


idx = ~np.isnan(npage_pconv)
popt, pcov = [], []
for k, i in enumerate(idx):
    popt_, pcov_ = curve_fit(sigmoid, npages[i], npage_pconv[k, i], p0=(15, 1))
    popt.append(popt_)
    pcov.append(pcov_)


## plot
fig4, axs4 = prob_distrib_plot(
    npages, npage_counts, npage_pconv, npage_std_pconv,
    group_labels=country)

axs4[0].text(14, 0.103, f'total counts: {int(np.sum(npage_counts))}')
axs4[0].set_ylim(0, 0.16)
axs4[0].set_title("# pages visited distribution")
axs4[0].set_xlabel('# pages visited')
axs4[0].legend(loc=1)

for k, popt_ in enumerate(popt):
    axs4[1].plot(npages, sigmoid(npages, *popt_),
                  linewidth=1, color=f'C{k}', label='sigmoid fit')
axs4[1].set_ylim(-0.005, 1.005)
axs4[1].set_xlabel('# pages visited')
# axs4[1].legend()

plt.show()

"""
Here we show, for each country, the distribution of pages visited (left panel) and the conversion probability as a function of the number of pages visited (right panel).
- The behavior of users from `'Germany'`, `'UK'` and `'US'` is quite similar both in terms of page visits and conversion probability thresholding with a tendency Germany < UK < US
which matches the differences observed in figure 1.
- The behavior difference of users from `'China'` is striking. First, the distibution of pages visits is different, with much less visits of more than 13 pages, those which are associated to newsletter subscription.
Second, the threshold seems to be set at a significantly higher level than other countries. These two factors explain the much lower conversion rates observed in figure 1.
"""


#%% 

### Combined effect of age and number of pages visited

new_user, npages = np.arange(80), np.arange(30)
country, counts, pconv, std_pconv = prob_distrib(
    df, group_by='new_user', variables=['age', 'total_pages_visited'],
    xvals=[ages, npages])


x, y = np.meshgrid(npages[:25], ages[17:46])

fig5, axs5 = plt.subplots(
    nrows=1, ncols=2,
    gridspec_kw={'left':0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1},
    subplot_kw=dict(projection='3d'))

# new visitors
axs5[0].view_init(elev=20, azim=-110)
surf = axs5[0].plot_surface(
    x, y, pconv[1, 17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs5[0].set_xlim(0, 24)
axs5[0].set_ylim(17, 45)
axs5[0].tick_params(pad=0)
axs5[0].set_title('New visitors')
axs5[0].set_xlabel('# pages visited')
axs5[0].set_ylabel('age')
axs5[0].set_zlabel('conv. prob.')

# recurring visitors
axs5[1].view_init(elev=20, azim=-110)
surf = axs5[1].plot_surface(
    x, y, pconv[0, 17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs5[1].set_xlim(0, 24)
axs5[1].set_ylim(17, 45)
axs5[1].tick_params(pad=0)
axs5[1].set_title('Recurring visitors')
axs5[1].set_xlabel('# pages visited')
axs5[1].set_ylabel('age')
axs5[1].set_zlabel('conv. prob.')

plt.show()

"""
The figure shows the conversion probability as a function of age and number of pages visited
for both new (left panel) and recurring (right panel) visitors.
- the sigmoid dependence of the conversion probability vs. number of pages visited is valid over the whole age range.
- In both conditions, there is a small linear dependence of the inflection point with age, with a lower inflection point for younger visitors.
"""

#%%
"""
## <a name="linear"></a>Logistic regression
"""

from sklearn.base import clone
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     cross_validate,
                                     StratifiedKFold,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer,
                                   PolynomialFeatures)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score,
                             confusion_matrix,
                             roc_curve,
                             auc)


def evaluate_model(pipeline: Pipeline,
                   export_csv: bool = False,
                   name_suffix: str = 'EXAMPLE')-> np.ndarray:
    """
    Evaluate the model on the test set. The predictions are exported as
    'conversion_data_test_predictions_{name_suffix}.csv'

    Parameters
    ----------
    Pipeline : Callable: pd.DataFrame -> T
        The feature preprocessor.
    export_csv : bool, optional
        Export the test data as a csv file for challenge assessment.
        The default is False.
    name_suffix : str, optional
        Suffix appended to the output filename. The default is 'EXAMPLE'.
    
    Returns
    -------
    Y_pred : 1D np.ndarray
        The model's predictions on the test set.
    """
    df = pd.read_csv('./conversion_data_test.csv')
    Y_pred = pipeline.predict(df)
    
    if export_csv:
        fname = f'conversion_data_test_predictions_{name_suffix}.csv'
        np.savetxt(fname, Y_pred, fmt='%d', header='converted', comments='')
    
    return Y_pred


# def evaluate_model(preprocessor: Callable,
#                    predictor: Callable,
#                    export_csv: bool = False,
#                    name_suffix: str = 'EXAMPLE')-> np.ndarray:
#     """
#     Evaluate the model on the test set. The predictions are exported as
#     'conversion_data_test_predictions_{name_suffix}.csv'

#     Parameters
#     ----------
#     preprocessor : Callable: pd.DataFrame -> T
#         The feature preprocessor.
#     prediction_engine : Callable: T -> np.ndarray
#         The prediciton engine of the model.
#     export_csv : bool, optional
#         Export the test data as a csv file for challenge assessment.
#         The default is False.
#     name_suffix : str, optional
#         Suffix appended to the output filename. The default is 'EXAMPLE'.
    
#     Returns
#     -------
#     Y_pred : 1D np.ndarray
#         The model's predictions on the test set.
#     """
#     df = pd.read_csv('./conversion_data_test.csv')
#     X = preprocessor(df)
#     Y_pred = predictor(X)
    
#     if export_csv:
#         fname = f'conversion_data_test_predictions_{name_suffix}.csv'
#         np.savetxt(fname, Y_pred, fmt='%d', header='converted', comments='')
    
#     return Y_pred


#%%
"""
### A first model

We reproduce here the basic linear regression from the template, this time including all the features in the fit.
We do a simple train-test split using the train set, without relying on the test set provided due to limited access.
"""

"""
Preprocessing and training
"""

# Data to fit
y = df.loc[:, 'converted']
X_df = df.drop('converted', axis=1)

# One-hot encoding with pandas
X_df = pd.get_dummies(X_df)
X = X_df.drop(['country_US', 'source_Seo'], axis=1)

# Simple train-test split
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# Data scaling: scale everything, including categorical data
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)

# fit logistic regression classifier
model1 = LogisticRegression()
model1.fit(X_tr, y_tr)


"""
Performance assessment
"""

# train set predicitions
y_tr_pred = model1.predict(X_tr)

# test set predicitions
X_test = scaler.transform(X_test) # no fit this time
y_test_pred = model1.predict(X_test)

# metrics
print(f'Train set F1-score: {f1_score(y_tr, y_tr_pred)}')
print("Train set confusion matrix\n",
      confusion_matrix(y_tr, y_tr_pred, normalize='all'))
print(f'Test set F1-score: {f1_score(y_test, y_test_pred)}')
print("Test set confusion matrix\n",
      confusion_matrix(y_test, y_test_pred, normalize='all'))


#%%

"""
### Improving the simple model

Let us now make some improvements to the previous code, without changing the model (almost).
"""

"""
A first improvement is to make it work in a more integrated fashion. We thus do the following:
- The data preprocessing, encoding of categorical variables and scaling of quantitative variables, is wrapped in a `compose.ColumnTransformer`
- The preprocessing and classification are wrapped in a `pipeline.Pipeline`, exposing a single interface for all steps

Note that one-hot encoding can be done with pandas
```python
X = pd.get_dummies(X).drop(['country_US', 'source_Seo'], axis=1)
```
"""

##
# data preparation
y = df.loc[:, 'converted']
X = df.drop('converted', axis=1)

# preprocessing
cat_vars = ['country', 'source']
bool_vars = ['new_user']
quant_vars = ['age', 'total_pages_visited']
col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop='first'), cat_vars),
     ('bool_id', FunctionTransformer(lambda x: x), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])

# full pipeline
pipeline = Pipeline([('column_preprocessing', col_preproc),
                     ('classifier', LogisticRegression())])

"""
The default logistic regression classifier uses a threshold probability of 0.5 to classify an observation.
This corresponds to the minimization of the logistic loss.
However, we recall that the performance of our model is assessed not by the average logistic loss but through the F1-score.
Our model can adjust by cross-validation this threshold probability so as to maximize the F1-score. This is done by wrapping the
pipeline into a `model_selection.TunedThresholdClassifierCV`.
"""

model1_ta = TunedThresholdClassifierCV(pipeline, scoring='f1', cv=10,
                                       random_state=1234,
                                       store_cv_results=True)
t0 = time.time()
model1_ta.fit(X, y)
t1 = time.time()
print(f'model fitting time: {t1-t0} s')

best_thr = model1_ta.best_threshold_
logit_thr = np.log(best_thr / (1 - best_thr))
best_score = model1_ta.best_score_

print(f'best threshold = {best_thr:.8f}',
      f'\nbest F1-score = {best_score:.8f}')

#%%
"""
## metrics: confusion matrix and ROC

The receiver operating characteristic (ROC) curve is the natural metric associated to the classification threshold variation.
We produce the mean curve through 10-times cross validation building a ROC curve from each validation set and averaging the results.

We also use the data to compute the mean confusion matrix at the threshold selected above. Compared to that of the non-adjusted linear regression,
we get 15% less false positives, at the expense of 50% more false negatives. However, since the latter are not involved in the computation of the F1 score,
the overall result is an improved value.
"""
n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
clf = Pipeline([('column_preprocessing', col_preproc),
                ('classifier', LogisticRegression())])

mean_cm = np.zeros((2, 2), dtype=float) # confusion matrix
mean_fpr = np.linspace(0, 1, 101) # false-positive rates
mean_tpr = np.zeros(101, dtype=float) # true-positive rates
thr_fpr, thr_tpr = 0, 0 # fpr, tpr at threshold
for i, (itr, ival) in enumerate(cv.split(X, y)):
    X_v, y_v = X.iloc[ival], y.iloc[ival]
    clf.fit(X.iloc[itr], y.iloc[itr])
    # confusion matrix
    y_pred = clf.predict_proba(X_v)[:, 1] > best_thr
    mean_cm += confusion_matrix(y_v, y_pred) / len(X)
    # ROC
    y_decision = clf.decision_function(X_v)
    fpr, tpr, thr = roc_curve(y_v, y_decision)
    mean_tpr += np.interp(mean_fpr, fpr, tpr, left=0, right=1) / n_splits
    thr_fpr += np.interp(logit_thr, thr[::-1], fpr[::-1]) / n_splits
    thr_tpr += np.interp(logit_thr, thr[::-1], tpr[::-1]) / n_splits
mean_tpr[0] = 0
mean_auc = auc(mean_fpr, mean_tpr)

print("Mean confusion matrix\n", mean_cm)


#%%
"""
We plot the ROC curve in figure 6. The location of our selected threshold is indicated on the curve as an orange cross.
"""

fig6, ax6 = plt.subplots(figsize=(5, 5))
fig6.suptitle('Figure 6', x=0.02, ha='left')

ax6.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})')
ax6.plot(thr_fpr, thr_tpr, marker='X', markersize=7, markeredgewidth=0.1)

ax6.legend(loc=4)
ax6.set_title('mean ROC curve')
ax6.set_xlabel('False positive rate')
ax6.set_ylabel('True positive rate')
ax6.set_xlim(-0.01, 1.01)
ax6.set_ylim(-0.01, 1.01)
ax6.grid(visible=True, alpha=0.3)

thr_txt = (f'threshold = {best_thr:.4f}\nF1-score = {best_score:.4f}\n'
           f'TPR = {thr_tpr:.4}\nFPR = {thr_fpr:.4}')
ax6.text(0.23, 0.74, thr_txt, ha='center', va='center')

plt.show()
sys.exit()

#%%

"""
As mentioned in the EDA section, there is an interaction of the variables. We can capture such interaction by introducing
polynomial features of the form $X_i X_j$. This is done with `preprocessing.PolynomialFeatures`. However, we limit ourselves
to 1st order interactions proper, and consider only quadratic *cross* terms $X_i X_j, \ i \neq j$.
Still, there are $n(n-1)/2 = 45$ such terms for $n=10$. This is a lot, and we expect many of those terms to have no predictive power.

In order to keep only relevant terms, we apply a lasso penalty. The penalty parameter 


we expect many cross terms X1*X2 to heve no effe
"""

# no need to drop values in OHE: we'll do regularization
col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
     ('bool_id', FunctionTransformer(lambda x: x), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])

# set the classifier
clf = LogisticRegression(
    penalty='l1', # 
    fit_intercept=False, # intercept already included as a feature by `PolynomialFeatures`
    solver='saga', # supports L1 penalty
    )

# full pipeline
pipeline = Pipeline(
    [('column_preprocessing', col_preproc),
     ('poly_features', PolynomialFeatures(degree=2, interaction_only=True)),
     ('classifier', clf)])

# optimize the L1 regularization parameter
gsearch = GridSearchCV(
    pipeline,  param_grid={'classifier__C': np.linspace(0.01, 0.1, num=21)},
    n_jobs=8, refit=False, cv=10, verbose=False)
gsearch.fit(X, y)
best_C = gsearch.best_params_['classifier__C']
print(f'found best C = {best_C}')
clf.C = best_C

# evaluate the model by cross validation
cv_model2 = cross_validate(pipeline, X, y, scoring='f1', cv=10)
model2_f1 = np.mean(cv_model2['test_score'])
model2_std_f1 = np.std(cv_model2['test_score'])
print(f'Mean F1-score = {model2_f1:.8} +- {model2_std_f1:.8}')

# fit on the complete dataset
model2 = pipeline.fit(X, y)

"""
The model is better than the simple linear regression. However, it is no better than the version with adjusted classification threshold.
We can proceed as before and adapt the classification threshold of this model to improve it further.
"""

model2_ta = TunedThresholdClassifierCV(
    clone(pipeline), scoring='f1', cv=10, random_state=1234)
t0 = time.time()
model2_ta.fit(X, y)
t1 = time.time()
print(f'model fitting time: {t1-t0} s')

best_thr = model2_ta.best_threshold_
logit_thr = np.log(best_thr / (1 - best_thr))
best_score = model2_ta.best_score_

print(f'best threshold = {best_thr:.8f}',
      f'\nbest F1-score = {best_score:.8f}')

"""
There is an improvement with respect to the non-adjusted model, but the score is not better than the adjusted simple linear regression.
"""

#%%

"""
The application of the `PolynomialFeatures` preprocessor caused the apparition of many useless features 
The many (and mostly useless) features added through application of the `PolynomialFeatures` preprocessor were the cause of a significant slowdown
of the various optimization procedures. 

A final approach, backed by intuition, is to consider each `'country'` and `'new_user'` value as leading to a different behavior. We can thus split the dataset into
multiple datasets, corresponding to a (country, new_user) pair, and fit a different model on each one. The predictions will then be made by dispatching to the appropriate model.


"""

dfs = {idx: df_ for idx, df_ in X.groupby(['country', 'new_user'])}




#%%
##### testing pipeline #####
from hashlib import shake_256

# def preprocessor_baseline(dataframe: pd.DataFrame)-> np.ndarray:
#     df = pd.get_dummies(dataframe)
#     df = df.drop(['country_US', 'source_Seo'], axis=1)
#     return df

# def predictor_baseline(X: np.ndarray)-> np.ndarray:
#     return model.predict(X)

models = {
    # 'model1_ta': model1_ta,
    'model2': model2,
    'model2_ta': model2_ta,
    
    }
# models = {m: shake_256(m).hexdigest(4) for m in models}

for m, model in models.items():
    evaluate_model(model, export_csv=True, name_suffix=m)

# evaluate_model(preprocessor_baseline,
#                predictor_baseline,
#                export_csv=False,
#                name_suffix=models[b'linreg_threhold_adjust'])



