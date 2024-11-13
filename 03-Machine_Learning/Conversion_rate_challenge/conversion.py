#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import time
import warnings

import numpy as np
import pandas as pd



# =============================================================================
# Data loading and initial preprocessing
# =============================================================================

# Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]


# test_df = pd.read_csv('./conversion_data_test.csv')
# test_df.describe(include='all')
# """
# The distribution of the different observations is essentially the same
# as that of the training set in terms of mean and standard deviation.
# """


# %% EDA
"""
## <a name="eda"></a>Preliminary EDA

We begin with a preliminary data analysis. Our problem is of low dimension, with only 5 predictive features and a binary categorical variable as the target.
A good approach in this case is to plot the average of the target as a function of all features.
Here this average can be interpreted naturally as the probability of subscription to the newsletter.
As we'll see, the insights gathered in this first step will point us towards more advanced data analysis.
"""

# %%% EDA utils
"""
### Utilities 

Before moving to the data exploration proper, we set-up a few utility functions to wrap around repetitive code:
- `prob_distrib` computes the newsletter subscription probability as a function of the selected variables.
- `prob_distrib_plot` produces a plot of the probability density of the selected variable and the corresponding subscription probability.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit


def prob_distrib(df: pd.DataFrame,
                 group_by: str,
                 variables: list[str],
                 xvals: list[np.ndarray],
                 ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each element in the `group_by` column, compute the conversion
    probability and a std deviation estimate as a function of the values in the
    columns in `variables`.
    xvals[i] contains an array of values taken by df[variables[i]].
    
    Returns
    -------
    group_vals : list
        The different elements in the `group_by` column of the dataframe.
    counts : np.ndarray of shape (ngroups,) + (len(x) for x in xvals)
        counts[k, i0, .., in] is the number of rows in the k-th group dataframe
        with variables values xvals[0][i0], ..., xvals[n][in]
    pconv : np.ndarray, same shape as `counts`
        The conversion probability (mean of column 'converted'). The value is
        NaN when the corresponding counts are zero.
    std_pconv : np.ndarray, same shape as `pconv`
        An estimate of the std deviation of pconv: p(1-p)/sqrt(n).

    """
    group_vals = []
    counts = np.zeros((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                      dtype=int)
    pconv = np.full((df[group_by].nunique(),) + tuple(len(x) for x in xvals),
                    np.nan, dtype=float)

    indices = [{x: i for i, x in enumerate(xs)} for xs in xvals] # manage the case of non-integer indices
    for k, (group, gdf) in enumerate(df.groupby(group_by)):
        group_vals.append(group)
        for idx, c in gdf.loc[:, variables].value_counts().items():
            idx = tuple(indice[i] for indice, i in zip(indices, idx))
            counts[k, *idx] = c
        pconv_df = gdf.loc[:, variables + ['converted']] \
                      .groupby(variables) \
                      .mean()['converted']
        for idx, p in pconv_df.items():
            idx = idx if isinstance(idx, tuple) else (idx,)  # cast to tuple if no multiindex
            idx = tuple(indice[i] for indice, i in zip(indices, idx))
            pconv[k, *idx] = p

    std_pconv = np.where(pconv * (1 - pconv) == 0., 1, pconv) / np.sqrt(counts)
    return group_vals, counts, pconv, std_pconv


def prob_distrib_plot(xvals: np.ndarray,
                      counts: np.ndarray,
                      pconv: np.ndarray,
                      std_pconv: np.ndarray,
                      group_labels: list[str]):
    """
    Make two plots:
    Left: using `counts`, plot the probablity density of the selected variable
          for each group.
    Right: using `pconv` and `std_pconv`, plot the conversion probability vs
           selected variable for each group.

    Parameters
    ----------
    xvals : 1D np.ndarray
        The values taken by the selected variable.
    counts, pconv, std_pconv : 2D np.ndarray, shape (ngroups, len(xvals))
        arr[i, j] = relevant quantity for group i, at xvals[j].
    group_labels : list[str]
        The group labels.

    Returns
    -------
    fig , axs: matplotlib Figure, np.ndarray[matplotlib Axes], shape (2,)
        The figure and axes, for further processing.

    """
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(9, 4),
        gridspec_kw={'left': 0.08, 'right': 0.96, 'top': 0.85, 'bottom': 0.12})

    axs[0].grid(visible=True)
    distribs = counts / np.sum(counts, axis=1, keepdims=True)
    for k, label in enumerate(group_labels):
        axs[0].plot(xvals, distribs[k], color=f'C{k}',
                    linewidth=1, label=f'{label} ({int(np.sum(counts[k]))})')
    axs[0].set_xlim(0, np.max(xvals)+1)
    axs[0].set_ylim(-0.0005, (int(np.max(distribs)*100)+1)/100)
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


# %%% EDA tables
"""
### Some tables

We begin by counting the conversion events vs the various categorical variables, `'country'`, `'new_user'`, `'source'`.
"""

##
df.groupby(['country', 'converted']).count()['age'] # the name 'age' is irrelevant here
##
df.groupby(['new_user', 'converted']).count()['age']
##
df.groupby(['source', 'converted']).count()['age']


"""
The conversion events are rather rare. About 1% for new website visitors and 8% for recurrent visitors, out of a global count of 300000.
Although the subscription probability is rather homogeneous accross the various sources, this is not the case for the different countries.
In particular, the conversion events are very rare among chinese visitors, with a count of only 89. We anticipate that this will cause a lot of variance in
classifying new events corresponding to chinese visitors.
"""


# %%% EDA heatmap
"""
### Conversion probability heatmap

We make the above observation more visual by showing heatmaps of the newsletter subscription probabilility for the
categorical features `'source'` and `'country'`, for both recurring (left) and new (right) visitors. We also indicate the conversion probability (in %)
the total counts corresponding to each condition.
"""

sources = ['Ads', 'Direct', 'Seo']
countries = ['China', 'Germany', 'UK', 'US']
new_user, counts, pconv, _ = prob_distrib(
    df, group_by='new_user', variables=['source', 'country'],
    xvals=[sources, countries])

##
fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, figsize=(6.5, 3.2),
    gridspec_kw={'left': 0.09, 'right': 0.94, 'top': 0.88, 'bottom': 0.11, 'wspace': 0.18})
fig1.suptitle('Figure 1: Influence of categorical variables on conversion probability', x=0.02, ha='left')

axs1_cb = np.empty_like(axs1, dtype=object)
for k, ax in enumerate(axs1):
    ax.set_aspect('equal')
    heatmap = ax.pcolormesh(100*pconv[k], cmap='plasma', edgecolors='k', lw=0.5)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], countries, rotation=60)
    ax.set_yticks([0.5, 1.5, 2.5], sources)
    for (i, j), p in np.ndenumerate(pconv[k]):
        color = 'k' if p > np.max(pconv[k]) / 3 else 'w'
        ax.text(j+0.5, i+0.5, f'{100*p:.2f}\n({counts[k, i, j]})',
                fontsize=9, color=color, ha='center', va='center')
    
    div = make_axes_locatable(ax)
    axs1_cb[k] = div.append_axes("right", size="7%", pad="5%")
    fig1.add_axes(axs1_cb[k])
    fig1.colorbar(heatmap, cax=axs1_cb[k], orientation="vertical", ticklocation="right")

axs1[0].set_title(f"Recurring visitors (n={np.sum(counts[0])})\nconversion prob. (%)")
axs1_cb[0].set_yticks([0, 5, 10, 15])

axs1[1].set_title(f"New visitors (n={np.sum(counts[1])})\nconversion prob. (%)")
axs1[1].tick_params(left=False, labelleft=False)
axs1_cb[1].set_yticks([0, 0.5, 1, 1.5, 2, 2.5])

plt.show()

"""
We note the following trends, that could already been seen from the raw data in the tables above:
- Recurring visitors are 2.5 - 8 times more likely to subscribe to the newsletter.
- Chinese visitors are much less likely to subscribe than others despite being the second population in terms of country.
- The way by which visitors reached the website has little impact on their eventual subscription.

We anticipate that the very low conversion probability among chinese users will cause a lot of variance in
the classification of new visits from China.
"""


# %%% EDA age
"""
### Influence of visitors' age on the conversion probability

We turn to the study of the impact of quantitative variables on the conversion probability, begining with the variable `'age'`.
"""

## Age data
ages = np.arange(80)
new_user, age_counts, age_pconv, age_std_pconv = prob_distrib(
    df, group_by='new_user', variables=['age'], xvals=[ages])

##
def decay_exp(x: np.ndarray,
              A: float,
              tau: float) -> np.ndarray:
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

## plot
fig2, axs2 = prob_distrib_plot(
    ages, age_counts, age_pconv, age_std_pconv,
    group_labels=['recur.', 'new'])
fig2.suptitle("Figure 2: Influence of 'age' on the conversion probability", x=0.02, ha='left')

axs2[0].text(45, 0.0375, f'total counts: {int(np.sum(age_counts))}')
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
We show in figure 2 the distribution of ages (left panel) and the conversion probability (right panel)
aggregated by `'source'` and `'country'`, but distinguishing new and recurrent visitors.
- The distribution of ages is the same for the two categories: a truncated gaussian with a cutoff at 17 years old,
  centered at around 28 years old. The probability of having a new visitor is thus roughly twice that of a recurrent visitor, independently of age.
- The newsletter subscription probability decays with age. The observed decay fits well with an exponential
  from which we get similar decay constants of about 14 years. We also recover the ratio of 5-6 of conversion probability between recurrent and new visitors.

Finally, we note that these patterns are extremely regular and again reveal the artificial nature of the data.
"""


# %%% EDA npages
# TODO comment
"""
### Influence of the number of pages visited

We proceed in a similar fashion this time with the quantity `'total_pages_visited'`.
"""
npages = np.arange(30)
new_user, npage_counts, npage_pconv, npage_std_pconv = prob_distrib(
    df, group_by='new_user', variables=['total_pages_visited'], xvals=[npages])


# Your beloved sigmoid fit
def sigmoid(x: np.ndarray,
            x0: float,
            a: float) -> np.ndarray:
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


# plot
fig3, axs3 = prob_distrib_plot(
    npages, npage_counts, npage_pconv, npage_std_pconv,
    group_labels=['recur. visitors', 'new visitors'])
fig3.suptitle("Figure 3: Influence of 'total_pages_visited' on the conversion probability",
              x=0.02, ha='left')

axs3[0].text(13, 0.123, f'total counts: {int(np.sum(npage_counts))}')
axs3[0].set_ylim(-0.002, 0.16)
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
axs3[1].set_title("Conversion probability")
axs3[1].set_xlabel('# pages visited')
axs3[1].set_ylabel('Conversion probability')
axs3[1].legend()

plt.show()


txt = """
The number of pages visited is clearly the most relevant feature that determines the newsletter subscription, with a sigmoid dependence of the conversion probability.
- The newsletter subscription probability looks like a sigmoid for both new and recurring visitors. The inflection points are shifted,
  and looking at the 13 pages-visits threshold, we see that here the conversion probability is thrice for recurring users than for new users.


- The distributions of number of pages visited are not proportional. They actually look like
  [inverse-gamma distributions](https://en.wikipedia.org/wiki/Inverse-gamma_distribution).

- Although we still have a factor ~2 between the total counts, we remark that website visits with more than 13 pages visited are
  mostly done by recurrent visitors. Accounting for the factor 2, this makes it more than twice likely that a > 13 pages visit
  is done by a recurrent visitor.


- Sigmoid fits with the slope and inflection point as parameters is also shown on the right panel. The slopes are roughly the same,
  the difference being the inflection point: 12.5 pages for recurring visitors vs 15 pages for new visitors.
- Finally, we note two outlier events on the right panel. ~30 pages visits with no subscription. We might consider removing these observations
  for training later on.

We again stress that such regularity is unlikely in any realistic context.
"""

# %%% country x npages
"""
### Effect of the number of pages visited for each country

We noted above that the country of origin has a significant impact on the conversion probability.
We consider the two following hypotheses to explain this trend:
- The country of orgin has an impact on the distribution of number of pages visited, but not on the subscription threshold (the inflection point of the sigmoid).
  In this case, the country is expected to not be a good a predictor: the relevant information would be already contained in the number of pages visited.
- The pages visits distribution does not depend on the country, but the conversion threshold does. In this case, the country (more precisely, its correlation with the number of pages visted)
  would be a highly relevant predictor.
For comparison, the new/recurring character of visitors effect is a mixture of these two: it affects both the pages visits distribution and conversion threshold.
"""

##
npages = np.arange(30)
country, npage_counts, npage_pconv, npage_std_pconv = prob_distrib(
    df, group_by='country', variables=['total_pages_visited'], xvals=[npages])

##
idx = ~np.isnan(npage_pconv)
popt, pcov = [], []
for k, i in enumerate(idx):
    popt_, pcov_ = curve_fit(sigmoid, npages[i], npage_pconv[k, i], p0=(15, 1))
    popt.append(popt_)
    pcov.append(pcov_)


### plot
fig4, axs4 = prob_distrib_plot(
    npages, npage_counts, npage_pconv, npage_std_pconv,
    group_labels=country)
fig4.suptitle("Figure 4: Cross-influence of 'country' and 'total_pages_visited'", x=0.02, ha='left')

axs4[0].text(14, 0.103, f'total counts: {int(np.sum(npage_counts))}')
axs4[0].set_ylim(-0.001, 0.16)
axs4[0].set_title("# pages visited distribution")
axs4[0].set_xlabel('# pages visited')
axs4[0].legend(loc=1)

for k, popt_ in enumerate(popt):
    axs4[1].plot(npages, sigmoid(npages, *popt_),
                 linewidth=1, color=f'C{k}', label='sigmoid fit')
axs4[1].set_ylim(-0.005, 1.005)
axs4[1].set_xlabel('# pages visited')

plt.show()

"""
Here we show, for each country, the distribution of pages visited (left panel) and the conversion probability as a function of the number of pages visited (right panel).
- The behavior of users from `'Germany'`, `'UK'` and `'US'` is quite similar both in terms of page visits and conversion probability thresholding with a tendency Germany < UK < US
which matches the differences observed in figure 1.
- The behavior difference of users from `'China'` is striking. First, the distibution of pages visits is different, with much less visits of more than 13 pages, those which are associated to newsletter subscription.
Second, the threshold seems to be set at a significantly higher level than other countries. These two factors explain the much lower conversion rates observed in figure 1.
"""

# %%% source x npages
# TODO comment
"""
### Effect of the number of pages visited for each source

We now consider 
"""

##
npages = np.arange(30)
source, npage_counts, npage_pconv, npage_std_pconv = prob_distrib(
    df, group_by='source', variables=['total_pages_visited'], xvals=[npages])

##
idx = ~np.isnan(npage_pconv)
popt, pcov = [], []
for k, i in enumerate(idx):
    popt_, pcov_ = curve_fit(sigmoid, npages[i], npage_pconv[k, i], p0=(15, 1))
    popt.append(popt_)
    pcov.append(pcov_)


## plot
fig5, axs5 = prob_distrib_plot(
    npages, npage_counts, npage_pconv, npage_std_pconv,
    group_labels=source)
fig5.suptitle("Figure 5: Cross-influence of 'source' and 'total_pages_visited' on the conversion probability",
              x=0.02, ha='left')

axs5[0].text(17, 0.11, f'total counts: {int(np.sum(npage_counts))}')
axs5[0].set_ylim(0, 0.16)
axs5[0].set_title("# pages visited distribution")
axs5[0].set_xlabel('# pages visited')
axs5[0].legend(loc=1)

for k, popt_ in enumerate(popt):
    axs5[1].plot(npages, sigmoid(npages, *popt_),
                 linewidth=1, color=f'C{k}', label='sigmoid fit')
axs5[1].set_ylim(-0.005, 1.005)
axs5[1].set_xlabel('# pages visited')
axs5[1].legend(loc=2)

plt.show()

"""
The plot shows...
"""


# %%% Age x npages
# TODO comment
"""
### Combined effect of age and number of pages visited
"""

new_user, npages = np.arange(80), np.arange(30)
country, counts, pconv, std_pconv = prob_distrib(
    df, group_by='new_user', variables=['age', 'total_pages_visited'],
    xvals=[ages, npages])


x, y = np.meshgrid(npages[:25], ages[17:46])

fig6, axs6 = plt.subplots(
    nrows=1, ncols=2,
    gridspec_kw={'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1},
    subplot_kw=dict(projection='3d'))
fig6.suptitle("Figure 6: Cross-influence of 'age' and 'total_pages_visited' on the conversion probability",
              x=0.02, ha='left')

# new visitors
axs6[0].view_init(elev=20, azim=-110)
surf = axs6[0].plot_surface(
    x, y, pconv[1, 17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs6[0].set_xlim(0, 24)
axs6[0].set_ylim(17, 45)
axs6[0].tick_params(pad=0)
axs6[0].set_title('New visitors')
axs6[0].set_xlabel('# pages visited')
axs6[0].set_ylabel('age')
axs6[0].set_zlabel('conv. prob.')

# recurring visitors
axs6[1].view_init(elev=20, azim=-110)
surf = axs6[1].plot_surface(
    x, y, pconv[0, 17:46, :25], rstride=1, cstride=1, cmap='coolwarm',
    linewidth=0, antialiased=True, shade=False)

axs6[1].set_xlim(0, 24)
axs6[1].set_ylim(17, 45)
axs6[1].tick_params(pad=0)
axs6[1].set_title('Recurring visitors')
axs6[1].set_xlabel('# pages visited')
axs6[1].set_ylabel('age')
axs6[1].set_zlabel('conv. prob.')

plt.show()

"""
The figure shows the conversion probability as a function of age and number of pages visited
for both new (left panel) and recurring (right panel) visitors.
- the sigmoid dependence of the conversion probability vs. number of pages visited is valid over the whole age range.
- In both conditions, there is a small linear dependence of the inflection point with age, with a lower inflection point for younger visitors.
"""

# %% General Utils
# TODO comment
"""
## <a name="utils"></a>Utilities
"""

from scipy.special import expit

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (confusion_matrix,
                             roc_curve,
                             auc)
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     StratifiedKFold,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   FunctionTransformer)


warnings.simplefilter("ignore", ConvergenceWarning)


# def poly_features(X: np.ndarray) -> np.ndarray:
#     """
#     Custom polynomial features construction:
#         - Original features, Xi
#         - 2nd order polynomials of quantitative features, Xi^2, Xi*Xj
#         - Products of categorical and quantitative features, Xi_cat * Xj_quant
#     """
#     X_ = np.empty((len(X), 26), dtype=float)
#     X_[:, :9] = X  # original features
#     X_[:, 9:11] = X[:, 7:9]**2  # age**2, npages**2
#     X_[:, 11] = X[:, 7] * X[:, 8]  # age * npages
#     X_[:, 12:19] = X[:, :7] * X[:, [7]]  # cat * age
#     X_[:, 19:26] = X[:, :7] * X[:, [8]]  # cat * npages
#     return X_


def print_metrics(cm: np.ndarray)-> None:
    """
    Print metrics related to the confusion matrix: precision, recall, F1-score.
    """
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1.
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1.
    print("Confusion matrix\n", cm / np.sum(cm))
    print(f'Precision: {prec:.8}; recall: {recall:.8}')
    print(f'F1-score: {2*prec*recall/(prec+recall):.8}')


def evaluate_model(model,
                   xtest_fname: str = 'conversion_data_test.csv',
                   ytest_fname: str = 'conversion_data_test_labels.csv'
                   )-> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test set. The predictions are exported as
    'conversion_data_test_predictions_{name_suffix}.csv'
    """
    X_test = pd.read_csv(xtest_fname)
    y_test = pd.read_csv(ytest_fname)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, cm


def cv_eval(model, X: np.ndarray, y: np.ndarray,
            n_splits: int = 10)-> np.ndarray:
    """
    Evaluation of a multi-model by cross-validation.
    Returns the confusion matrix.
    """
    cm = np.zeros((2, 2), dtype=int)  # confusion matrix
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (itr, ival) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[itr], y.iloc[itr])
        cm += confusion_matrix(y.iloc[ival], model.predict(X.iloc[ival]))
    return cm


def gridsearch_cv(model, X: np.ndarray, y: np.ndarray,
                  param_name: str, param_vals: np.ndarray,
                  n_splits: int = 10)-> float:
    """
    Grid search parameter optimization of a model by cross-validation.
    The parameters must be in the 'classifier' step of the model.
    """
    gsearch = GridSearchCV(
        model,  param_grid={f'classifier__{param_name}': param_vals},
        scoring='f1', n_jobs=4, refit=False, cv=n_splits)
    gsearch.fit(X, y)
    best_param = gsearch.best_params_[f'classifier__{param_name}']
    print(f'found best {param_name}: {best_param}')
    return best_param


def tune_threshold_cv(model, X: np.ndarray, y: np.ndarray,
                      n_splits: int = 10,
                      verbose: bool = True):
    """
    Tune the decision threhold of the model to maximize the F1 score by 
    cross-validation.
    """
    model_ta = TunedThresholdClassifierCV(
        model, scoring='f1', cv=n_splits, refit=True, random_state=1234)
    t0 = time.time()
    model_ta.fit(X, y)
    t1 = time.time()
    
    best_thr = model_ta.best_threshold_
    best_score = model_ta.best_score_
    
    if verbose:
        print(f'model fitting time: {t1-t0} s')
        print(f'best threshold = {best_thr:.8f}\nbest F1-score = {best_score:.8f}')
    
    return model_ta

"""
Some utilities

"""


def split_data(groups: list[str])-> tuple[dict, dict]:
    """
    Split the data as a dict : group_index -> dataframe.
    """
    Xs = {idx: df_.drop(groups, axis=1) for idx, df_ in X.groupby(groups)}
    ys = {idx: y.loc[X_.index] for idx, X_ in Xs.items()}
    return Xs, ys


def multimodel_gridsearch_cv(models: dict, Xs: dict, ys: dict,
                             param_name: str, param_vals: np.ndarray,
                             n_splits: int = 10)-> np.ndarray:
    """
    Grid search parameter optimization of a multi-model by cross-validation.
    The `models`, `Xs` and `ys` are dict : group_index -> model/data.
    
    The parameters must be in the 'classifier' step of the model.
    """
    best_params = {}
    for idx, model in models.items():
        print(f'{idx} ::', end='')
        best_params[idx] = gridsearch_cv(
            model, Xs[idx], ys[idx], param_name, param_vals, n_splits)
        
    return best_params


def multimodel_cv_eval(models: dict, Xs: dict, ys: dict,
                       n_splits: int = 10,
                       print_partial_scores: bool = False
                       )-> tuple[dict, np.ndarray]:
    """
    Evaluation of a multi-model by cross-validation.
    The `models`, `Xs` and `ys` are dict : group_index -> model/data.
    """
    cms = {}
    for idx, model in models.items():
        X_, y_ = Xs[idx], ys[idx]
        cms[idx] = cv_eval(model, X_, y_, n_splits)
    cm = sum(cm_ for cm_ in cms.values()) # confusion matrix
    
    if print_partial_scores:
        print('===== Partial scores =====')
        for idx, cm_ in cms.items():
            print(f'== index: {idx} ==')
            print_metrics(cm_)
    print('\n===== Global scores =====')
    print_metrics(cm)
    
    return cms, cm


class MimicEstimator():
    """
    Wrapper around a dict of models providing a `predict` for model evaluation
    on test data.
    """

    def __init__(self, model: dict, groups: list[str]):
        self.groups: list[str] = groups
        self.model: dict = model

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        y_pred = np.empty(len(df), dtype=int)
        Xs = {idx: df_.drop(groups, axis=1)
              for idx, df_ in df.groupby(self.groups)}
        for idx, X in Xs.items():
            y_pred[X.index] = self.model[idx].predict(X)
        return y_pred


# %% LogReg
# TODO comment
"""
## <a name="linear"></a>Logistic regression


"""

from sklearn.linear_model import LogisticRegression


# %%% LR : basic
# TODO comment
"""
### A first model

We reproduce here the basic linear regression from the template, this time including all the features in the fit.
We do a simple train-test split using the train set, without relying on the test set provided due to limited access at the time of writing.


#### Model construction and training
"""

# Data to fit
y = df.loc[:, 'converted']
X_df = df.drop('converted', axis=1)

# One-hot encoding with pandas
X_df = pd.get_dummies(X_df)
X = X_df.drop(['country_US', 'source_Seo'], axis=1)

# Simple train-test split
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1234)

# Data scaling: scale everything, including categorical data
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)

# fit logistic regression classifier
lr_model = LogisticRegression()
lr_model.fit(X_tr, y_tr)


"""
#### Performance assessment
"""

print('===== Train set metrics =====')
y_tr_pred = lr_model.predict(X_tr)
print_metrics(confusion_matrix(y_tr, y_tr_pred))

print('\n===== Test set metrics =====')
X_test = scaler.transform(X_test)  # no fit this time
y_test_pred = lr_model.predict(X_test)
print_metrics(confusion_matrix(y_test, y_test_pred))

"""

"""

t0 = time.time()
lr_model.fit(X, y)
t1 = time.time()
print(f'model fitting time: {t1-t0} s')


# %%% LR : threshold adjust
# TODO comment
"""
### Adjusting the decision threshold

Let us now make some improvements to the previous code, without changing the model (almost).

#### Model construction and training

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
     ('bool_id', FunctionTransformer(None), bool_vars),
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

lr_model_ta = TunedThresholdClassifierCV(
    pipeline, scoring='f1', cv=10, random_state=1234)

t0 = time.time()
lr_model_ta.fit(X, y)
t1 = time.time()
print(f'model fitting time: {t1-t0} s')

best_thr = lr_model_ta.best_threshold_
best_score = lr_model_ta.best_score_
print(f'best threshold = {best_thr:.8f}',
      f'\nbest F1-score = {best_score:.8f}')


# %%% LR : ROC
# TODO comment
"""
### Receiver operating charcteristic (ROC) curve

The receiver operating characteristic curve is the natural metric associated to the classification threshold variation.
We produce the mean curve through 10-times cross validation building a ROC curve from each validation set and averaging the results.

We also use the data to compute the mean confusion matrix at the threshold selected above. Compared to that of the non-adjusted linear regression,
we get 15% less false negatives, at the expense of 50% more false positives. However,

 since the latter are not involved in the computation of the F1 score,
the overall result is an improved value.
"""
n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
clf = Pipeline([('column_preprocessing', col_preproc),
                ('classifier', LogisticRegression())])

mean_cm = np.zeros((2, 2), dtype=float)  # confusion matrix
mean_fpr = np.linspace(0, 1, 401)  # false-positive rates
mean_tpr = np.zeros(401, dtype=float)  # true-positive rates
thr_fpr, thr_tpr = 0, 0  # fpr, tpr at threshold
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
    thr_fpr += np.interp(best_thr, expit(thr[::-1]), fpr[::-1]) / n_splits
    thr_tpr += np.interp(best_thr, expit(thr[::-1]), tpr[::-1]) / n_splits
mean_tpr[0] = 0
mean_auc = auc(mean_fpr, mean_tpr)


print('===== CV-estimated metrics =====')
print_metrics(mean_cm)


"""
We plot the ROC curve in figure 7. The location of our selected threshold is indicated on the curve as an orange cross.
"""

fig7, ax7 = plt.subplots(figsize=(5, 5))
fig7.suptitle('Figure 7: ROC curve', x=0.02, ha='left')

ax7.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})')
ax7.plot(thr_fpr, thr_tpr, marker='X', markersize=7, markeredgewidth=0.1)

ax7.legend(loc=4)
ax7.set_title('mean ROC curve')
ax7.set_xlabel('False positive rate')
ax7.set_ylabel('True positive rate')
ax7.set_xlim(-0.01, 1.01)
ax7.set_ylim(-0.01, 1.01)
ax7.grid(visible=True, alpha=0.3)

thr_txt = (f'threshold = {best_thr:.4f}\nF1-score = {best_score:.4f}\n'
           f'TPR = {thr_tpr:.4}\nFPR = {thr_fpr:.4}')
ax7.text(0.23, 0.74, thr_txt, ha='center', va='center')

plt.show()


"""
... comment
"""

# %%% LR : polynomial features
# TODO comment
"""
### Adding polynomial features


The application of the `PolynomialFeatures` preprocessor caused the apparition of many useless features 
The many (and mostly useless) features added through application of the `PolynomialFeatures` preprocessor were the cause of a significant slowdown
of the various optimization procedures. 

A final approach, backed by intuition, is to consider each `'country'` and `'new_user'` value as leading to a different behavior. We can thus split the dataset into
multiple datasets, corresponding to a (country, new_user) pair, and fit a different model on each one. The predictions will then be made by dispatching to the appropriate model.


#### Model construction

"""

# preprocessing
col_preproc_full = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
     ('bool_id', FunctionTransformer(None), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])

# custom feature engineering
def poly_features_full(X: np.ndarray) -> np.ndarray:
    """
    Custom polynomial features construction:
        - Original features, Xi
        - 2nd order polynomials of quantitative features, Xi^2, Xi*Xj
        - Products of categorical and quantitative features, Xi_cat * Xj_quant
        
    This function is adapted to the *full* dataset preprocessed with one-hot
    encoding of the categorical variables 'country' and 'source'.
    """
    X_ = np.empty((len(X), 29), dtype=float)
    X_[:, :10] = X  # original features
    X_[:, 10:12] = X[:, 8:10]**2  # age**2, npages**2
    X_[:, 12] = X[:, -2] * X[:, -1]  # age * npages
    X_[:, 13:21] = X[:, :8] * X[:, [8]]  # cat * age
    X_[:, 21:29] = X[:, :8] * X[:, [9]]  # cat * npages
    return X_

# classifier
lr = LogisticRegression(
    penalty='elasticnet',
    C=0.1,
    fit_intercept=True,
    class_weight=None,
    solver='saga',  # supports elasticnet penalty
    random_state=1234,
    l1_ratio=0.8)

# full pipeline
pipeline = Pipeline([('column_preprocessing', col_preproc_full),
                     ('poly_features', FunctionTransformer(poly_features_full)),
                     ('classifier', lr)])

# evaluate by cross-validation
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# adjujst threshold
lr_model_polyfeatures_ta = tune_threshold_cv(pipeline, X, y)


# %%% LR : split
# TODO comment
"""
### Splitting the model


The application of the `PolynomialFeatures` preprocessor caused the apparition of many useless features 
The many (and mostly useless) features added through application of the `PolynomialFeatures` preprocessor were the cause of a significant slowdown
of the various optimization procedures. 

A final approach, backed by intuition, is to consider each `'country'` and `'new_user'` value as leading to a different behavior. We can thus split the dataset into
multiple datasets, corresponding to a (country, new_user) pair, and fit a different model on each one. The predictions will then be made by dispatching to the appropriate model.


#### Model construction

"""

# split the dataset
groups = ['new_user', 'country']
Xs, ys = split_data(groups)

# column preprocessing
split_cat_vars = [v for v in cat_vars if v not in groups]
split_bool_vars = [v for v in bool_vars if v not in groups]
split_quant_vars = [v for v in quant_vars if v not in groups]
col_preproc_split = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), split_cat_vars),
     ('bool_id', FunctionTransformer(None), split_bool_vars),
     ('quant_scaler', StandardScaler(), split_quant_vars)])


# custom feature engineering
def poly_features_split(X: np.ndarray) -> np.ndarray:
    """
    Custom polynomial features construction:
        - Original features, Xi
        - 2nd order polynomials of quantitative features, Xi^2, Xi*Xj
        - Products of categorical and quantitative features, Xi_cat * Xj_quant
        
    This function is adapted to datasets *split* according to 'new_user' and
    'country', preprocessed with one-hot encoding of the remaining categorical
    variable 'source'.
    """
    X_ = np.empty((len(X), 13), dtype=float)
    X_[:, :5] = X  # original features
    X_[:, 5:7] = X[:, 3:5]**2  # age**2, npages**2
    X_[:, 7] = X[:, 3] * X[:, 4]  # age * npages
    X_[:, 7:10] = X[:, :3] * X[:, [3]]  # cat * age
    X_[:, 10:13] = X[:, :3] * X[:, [4]]  # cat * npages
    return X_


# classifier
# lr_split = LogisticRegression(
#     penalty='elasticnet',
#     C=0.1,
#     fit_intercept=True,
#     class_weight=None,
#     solver='saga',  # supports elasticnet penalty
#     random_state=1234,
#     l1_ratio=0.8,
# )
lr_split = LogisticRegression(
    penalty='l1',
    C=0.1,
    fit_intercept=True,
    class_weight=None,
    solver='saga',  # supports elasticnet penalty
    random_state=1234,
    l1_ratio=0.8,
)

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_split),
     ('poly_features', FunctionTransformer(poly_features_split)),
     ('classifier', lr_split)]
)
pipelines = {idx: clone(pipeline) for idx in Xs}

# multimodel_gridsearch_cv(pipelines, Xs, ys,
#                          param_name: str, param_vals: np.ndarray)


"""
#### Model evaluation by cross-validation
"""
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)



"""
#### Fitting the model and adjusting the threshold

Similarly, we can adjust the classification threshold
It would be difficult to recover a global F1-score estimate since the TunedThresholdClassifierCV won't
return the confusion matrix at each CV round.
"""

split_lr_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_lr_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)



# %%% LR : test eval

"""
### Model evaluation on test data

Our two models are `dict`s and as such do not have the `Estimator` interface (and most notably the `predict` method).
To perform evaluation, we wrap our model in a class mimicking the sufficient `Estimator` interface for model evaluation on the test data.
"""

##
# print('===== Basic logistic regression =====')
# _, cm = evaluate_model(lr_model)
# print_metrics(cm)

##
# print('\n===== Logistic regression with adjusted decision threshold =====')
# _, cm = evaluate_model(lr_model_ta)
# print_metrics(cm)


##
# print('===== Improved logistic regression =====')
# _, cm = evaluate_model(MimicEstimator(lr_model))
# print_metrics(cm)

##
# print('\n===== Improved logistic regression with adjusted classification threshold =====')
# _, cm = evaluate_model(MimicEstimator(lr_model_ta))
# print_metrics(cm)



# %% LDA
# TODO comment
"""
## <a name="lda"></a>Linear discriminant analysis

Problem with KNN: no metric, all features have same relevance.
Solve with `neighbors.NeighborhoodComponentsAnalysis`

"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# %%% LDA : Simple
# TODO comment
"""
### Simple model
"""

# classifier
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


# pipeline
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_full),
     ('poly_features', FunctionTransformer(poly_features_full)),
     ('classifier', lda)]
)

# optimization
best_shrinkage = gridsearch_cv(
    pipeline, X, y, param_name='shrinkage', param_vals=np.logspace(-3, -2, 41))
pipeline['classifier'].shrinkage = best_shrinkage


# assessment
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# fit
lda_model_ta = tune_threshold_cv(pipeline, X, y)


# %%% LDA : split
# TODO comment
"""

"""

# classifier
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# full pipelines
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_split),
     ('classifier', lda)]
)
pipelines = {idx: clone(pipeline) for idx in Xs}

# optimize regularization
best_shrinkages = multimodel_gridsearch_cv(
    pipelines, Xs, ys, param_name='shrinkage', param_vals=np.logspace(-3, 0, 31))
for idx, pipeline in pipelines.items():
    pipeline['classifier'].shrinkage = best_shrinkages[idx]

## CV-evaluation
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)


"""
... comment
"""


# Threshold adjust
split_lda_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_lda_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)



# %%% LDA : eval

"""

"""

# print('\n===== Linear discriminant analysis =====')
# _, cm = evaluate_model(lda_model)
# print_metrics(cm)

# print('\n===== Linear discriminant analysis with adjusted classification threshold =====')
# _, cm = evaluate_model(lda_model_ta)
# print_metrics(cm)

"""
... comment...
"""


# %% Trees
# TODO comment
"""
## <a name="tree"></a>Decision trees

"""

from sklearn.tree import DecisionTreeClassifier

# %%% Tree : simple
# TODO comment
"""
entropy/gini/log_loss => no significant change

for such a simple problem, pruning the tree is not necessary

Trees are insensitive to variables scaling
"""

# column preprocessing
tree_col_preproc_full = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), cat_vars),
     ('id_', FunctionTransformer(None), bool_vars + quant_vars)])

# classifier
dtc = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=1234)


# full pipeline, including polynomial feature generation
pipeline = Pipeline([('column_preprocessing', col_preproc),
                     ('classifier', dtc)])

# optimize
best_max_depth = gridsearch_cv(pipeline, X, y, param_name='max_depth',
                               param_vals=np.arange(2, 16, 1))
pipeline['classifier'].max_depth = best_max_depth

# model ssessment
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# fit
dtc_model_ta = tune_threshold_cv(pipeline, X, y, verbose=True)
# t0 = time.time()
# dtc_model.fit(X, y)
# t1 = time.time()
# print(f'model fitting time: {t1-t0} s')


# %%% Tree : polynomial features

"""

"""
# TODO comment
# feature engineering
def poly_features(X: np.ndarray) -> np.ndarray:
    """
    Custom polynomial features construction for ordinal-encoded categorical
    variables:
        - Original features, Xi
        - 2nd order polynomials of quantitative features, Xi^2, Xi*Xj
        - Products of categorical and quantitative features, Xi_cat * Xj_quant
    """
    X_ = np.empty((len(X), 24), dtype=float)
    X_[:, :5] = X  # original features
    X_[:, 5:7] = X[:, -2:]**2  # age**2, npages**2
    X_[:, 7] = X[:, -2] * X[:, -1]  # age * npages
    for i in range(4): # 4 countries
        X_[:, 8+i] = (X[:, 0] == i) * X[:, -2] # country * age
        X_[:, 16+i] = (X[:, 0] == i) * X[:, -1] # country * npages
    for i in range(3): # 3 sources
        X_[:, 12+i] = (X[:, 1] == i) * X[:, -2] # source * age
        X_[:, 20+i] = (X[:, 1] == i) * X[:, -1] # source * npages
    X_[:, 15] = X[:, 2] * X[:, -2] # new_user * age
    X_[:, 23] = X[:, 2] * X[:, -1] # new_user * npages
    return X_

# classifier
dtc = DecisionTreeClassifier(criterion='gini',
                             max_depth=best_max_depth,
                             random_state=1234)

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', tree_col_preproc_full),
     ('poly_features', FunctionTransformer(poly_features)),
     ('classifier', dtc)]
)

# 
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

"""
Adding polynomial features does not improve significantly the score, as expected since trees naturally capture the non-linear relationship between features.
"""

# %%% Tree : split
# TODO comment
"""
Split 
"""

# column preprocessing
tree_col_preproc_split = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), split_cat_vars),
     ('id_', FunctionTransformer(None), split_bool_vars + split_quant_vars)])

# full pipeline
pipeline = Pipeline([('column_preprocessing', tree_col_preproc_split),
                     ('classifier', dtc)])
pipelines = {idx: clone(pipeline) for idx in Xs}

## CV-evaluation
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)


"""
... comment
"""

# Threshold adjust
split_dtc_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_dtc_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)


# %%% Tree : test eval

"""
### Model evaluation on test data
"""

# print('===== Decision tree with adjusted decision threshold =====')
# _, cm = evaluate_model(dtc_model_ta)
# print_metrics(cm)


# print('\n===== Split decision tree with adjusted threshold =====')
# _, cm = evaluate_model(MimicEstimator(split_dtc_model_ta))
# print_metrics(cm)


# %% SVM
# TODO comment
"""
## <a name="svm"></a>Support Vector Machines

As an illustration of kernel methods for classification, we build a support vector classifier.

"""

from sklearn.svm import LinearSVC


# %%% SVM : simple
# TODO comment
"""
### Support Vector Machines

We saw that the main issue with the simple logistic regression was the strong imbalance between the conversion rates, and the fact that
the logistic regression would include all observations in the error function computation.

The support vector machine brings a solution to this issue, by considering only those observations that lie within a given margin of the decision plane.
Let us implement this solution.

No improvement from polynomial features, but convergence times x25
"""

# classifier
svc = LinearSVC(penalty='l2',
                loss='squared_hinge',
                C=1,
                fit_intercept=True,
                intercept_scaling=10,
                random_state=1234)

# full pipeline
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_full),
     ('classifier', svc)]
)

# optimize
best_C = gridsearch_cv(pipeline, X, y, param_name='C',
                       param_vals=np.logspace(-2, 1, 16))
pipeline['classifier'].C = best_C

# model assessment
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# fit
svc_model_ta = tune_threshold_cv(pipeline, X, y, verbose=True)


# %%% SVM : split
# TODO comment
"""
### Split SVM

Let us make a composite model
"""

# classifier
svc = LinearSVC(penalty='l2',
                loss='squared_hinge',
                C=best_C,
                fit_intercept=True,
                intercept_scaling=10,
                random_state=1234)

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_split),
     # ('poly_features', FunctionTransformer(poly_features_split)),
     ('classifier', svc)]
)
pipelines = {idx: clone(pipeline) for idx in Xs}

# assessment
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)

# Threshold adjust
split_svc_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_svc_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)


# %%% SVM : test eval

"""
### Model evaluation on test data
"""

# print('===== Support vector classifier with adjusted decision threshold =====')
# _, cm = evaluate_model(svc_model_ta)
# print_metrics(cm)


# print('\n===== Split support vector classifier with adjusted threshold =====')
# _, cm = evaluate_model(MimicEstimator(split_svc_model_ta))
# print_metrics(cm)


# %% NN
# TODO comment
"""
## <a name="nn"></a>Neural networks

"""
from sklearn.neural_network import MLPClassifier


# %%% NN : simple
# TODO comment
# TODO
"""
### Simple multi-layer perceptron
"""
# # column preprocessing
# col_preproc = ColumnTransformer(
#     [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
#      ('bool_id', FunctionTransformer(None), bool_vars),
#      ('quant_scaler', StandardScaler(), quant_vars)])

# # feature engineering
# def poly_features(X: np.ndarray) -> np.ndarray:
#     """
#     Custom polynomial features construction:
#         - Original features, Xi
#         - 2nd order polynomials of quantitative features, Xi^2, Xi*Xj
#         - Products of categorical and quantitative features, Xi_cat * Xj_quant
#     """
#     X_ = np.empty((len(X), 29), dtype=float)
#     X_[:, :10] = X  # original features
#     X_[:, 10:12] = X[:, 8:10]**2  # age**2, npages**2
#     X_[:, 12] = X[:, -2] * X[:, -1]  # age * npages
#     X_[:, 13:21] = X[:, :8] * X[:, [8]]  # cat * age
#     X_[:, 21:29] = X[:, :8] * X[:, [9]]  # cat * npages
#     return X_

# classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 20),
                    activation='relu',
                    batch_size=2000,
                    random_state=1234,
                    warm_start=True,
                    early_stopping=True)


# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', col_preproc),
     # ('poly_features', FunctionTransformer(poly_features)),
     ('classifier', mlp)]
)


# TODO train test split for eval
# _ = cv_eval(pipeline, X, y, verbose=True)

mlp_model_ta = tune_threshold_cv(pipeline, X, y, verbose=True)


# %%% NN : split
# TODO comment
"""
### Split model
"""

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', col_preproc_split),
     # ('poly_features', FunctionTransformer(poly_features_split)),
     ('classifier', mlp)]
)
pipelines = {idx: clone(pipeline) for idx in Xs}

# assessment
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)

# Threshold adjust
split_mlp_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_svc_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)




# %% Random forests
# TODO comment
"""
## <a name="rf"></a>Random forests

We now turn to ensemble methods.
"""

from sklearn.ensemble import RandomForestClassifier


# %%% RF : Simple model
# TODO comment
"""
### Simple model
"""

# classifier
rfc = RandomForestClassifier(criterion='gini',
                             # max_depth=8,
                             bootstrap=True,
                             random_state=1234)

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', tree_col_preproc_full),
     ('classifier', rfc)]
)

# optimize
best_max_depth = gridsearch_cv(pipeline, X, y, param_name='max_depth',
                               param_vals=np.arange(2, 13, 1))
pipeline['classifier'].max_depth = best_max_depth

# assessment
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# fit
rfc_model_ta = tune_threshold_cv(pipeline, X, y, verbose=True)


# %%% RF : Split model
# TODO
# TODO comment
"""
### Split model

"""

# classifier
rfc = RandomForestClassifier(criterion='gini',
                             max_depth=best_max_depth,
                             bootstrap=True,
                             random_state=1234)

# full pipeline
pipeline = Pipeline([('column_preprocessing', tree_col_preproc_split),
                     ('classifier', rfc)])
pipelines = {idx: clone(pipeline) for idx in Xs}

## CV-evaluation
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)

# Threshold adjust
split_rfc_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_rfc_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)


# %%% Tree : test eval

"""
### Model evaluation on test data
"""

# print('===== Random forest with adjusted decision threshold =====')
# _, cm = evaluate_model(rfc_model_ta)
# print_metrics(cm)


# print('\n===== Split random forests with adjusted threshold =====')
# _, cm = evaluate_model(MimicEstimator(split_rfc_model_ta))
# print_metrics(cm)


# %% Gradient boosting
# TODO comment
"""
## <a name="gb"></a>Gradient boosting
"""

from sklearn.ensemble import HistGradientBoostingClassifier


# %%% GB : Simple model
# TODO comment
"""
### Simple model

In order to have reasonable fitting times, we use the hisogram-based gradient boosting classifier
"""

# classifier
hgbc = HistGradientBoostingClassifier(loss='log_loss',
                                      max_iter=200,
                                      # max_depth=6,
                                      categorical_features=[0, 1, 2],
                                      random_state=1234)

# full pipeline, including polynomial feature generation
pipeline = Pipeline(
    [('column_preprocessing', tree_col_preproc_full),
     ('classifier', hgbc)]
)

# optimize
best_max_depth = gridsearch_cv(pipeline, X, y, param_name='max_depth',
                               param_vals=np.arange(2, 13, 1))
pipeline['classifier'].max_depth = best_max_depth

# assessment
cm = cv_eval(pipeline, X, y)
print_metrics(cm)

# fit
hgbc_model_ta = tune_threshold_cv(pipeline, X, y, verbose=True)


# %%% GB : Split model
# TODO comment
"""
### Split model

"""

# classifier
hgbc = HistGradientBoostingClassifier(loss='log_loss',
                                      max_iter=200,
                                      max_depth=best_max_depth,
                                      categorical_features=[0, 1, 2],
                                      random_state=1234)

# full pipeline
pipeline = Pipeline([('column_preprocessing', tree_col_preproc_split),
                     ('classifier', hgbc)])
pipelines = {idx: clone(pipeline) for idx in Xs}

## CV-evaluation
_ = multimodel_cv_eval(pipelines, Xs, ys, print_partial_scores=True)

# Threshold adjust
split_hgbc_model_ta = {}
for idx, p in pipelines.items():
    print(f'== {idx} ==')
    split_hgbc_model_ta[idx] = tune_threshold_cv(p, Xs[idx], ys[idx], verbose=True)


# %%% Tree : test eval

"""
### Model evaluation on test data
"""

# print('===== Gradient boosting with adjusted decision threshold =====')
# _, cm = evaluate_model(hgbc_model_ta)
# print_metrics(cm)


# print('\n===== Split gradient boosting with adjusted threshold =====')
# _, cm = evaluate_model(MimicEstimator(split_hgbc_model_ta))
# print_metrics(cm)


# %% Composite
"""
## <a name="composite"></a>Composite models

We have build many models so far, each having a F1 score of 0.75 - 0.77 after parameter tuning.
However, we saw that there were significant differences between models in terms of precision/recall to yield the F1 score, as illustrated in the following table.
"""

models = {
    # logistic regressions
    'lr_model_ta': lr_model_ta,
    'lr_model_polyfeatures_ta': lr_model_polyfeatures_ta,
    'split_lr_model_ta': split_lr_model_ta,
    # linear discriminant analysis
    'lda_model_ta': lda_model_ta,
    'split_lda_model_ta': split_lda_model_ta,
    # Decision trees
    'dtc_model_ta': dtc_model_ta,
    'split_dtc_model_ta': split_dtc_model_ta,
    # support vector machines
    'svc_model_ta': svc_model_ta,
    'split_svc_model_ta': split_svc_model_ta,
    # multi-layer perceptrons
    'mlp_model_ta': mlp_model_ta,
    'split_mlp_model_ta': split_mlp_model_ta,
    # random forests
    'rfc_model_ta': rfc_model_ta,
    'split_rfc_model_ta': split_rfc_model_ta,
    # gradient boosting
    'hgbc_model_ta': hgbc_model_ta,
    'split_hgbc_model_ta': split_hgbc_model_ta,
    }

# prec | recall | F1 of the models


"""
Furthermore, when fitting models for each pair (new_user, country)it appeared that the F1 scores were rather low in some categories and high in others,
as highlighted in the following table.
"""

# F1 for each idx of each model

"""
In this section, we will try to take advantage of each model by building composite models.
"""

from sklearn.ensemble import VotingClassifier

# %%% Voting

"""
### Voting classifier

We build a voting classifier out of all our models
"""

# vc = VotingClassifier()


# %%% Mixing models

"""
### Mixing models

We take the best of each sub-model
"""

