# -*- coding: utf-8 -*-
"""
!!!
# Lu matin
# 3 choses positives
# agressions dans le metro
# Celia et pierre baisent pas assez
Created on Fri Nov  8 10:49:22 2024

@author: netes
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit

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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


# =============================================================================
# Utils
# =============================================================================

def print_metrics(cm: np.ndarray)-> None:
    """
    Print metrics related to the confusion matrix: precision, recall, F1-score.
    """
    t = np.sum(cm, axis=1)[1]
    recall = (cm[1, 1] / t) if t != 0 else 1
    t = np.sum(cm, axis=0)[1]
    prec = (cm[1, 1] / t) if t != 0 else 1
    print("Confusion matrix\n", cm / np.sum(cm))
    print(f'Precision: {prec:.8}; recall: {recall:.8}')
    print(f'F1-score: {2*prec*recall/(prec+recall):.8}')


def tune_thr_cv(model, X: pd.DataFrame, y: pd.DataFrame,
                thrs: np.ndarray = np.linspace(0, 1, 101),
                n_splits: int = 10)-> float:
    """
    Tune the classification threshold according to the F1-score.
    X and y correspond to raw data, to be classified,
    """
    cms = np.zeros((2, 2, len(thrs)))
    thrs = np.reshape(thrs, (-1, 1))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (itr, ival) in enumerate(cv.split(X, y)):
        # fit
        df = X.iloc[itr]
        df.loc[:, 'pconv'] = y.iloc[itr]
        df = df.groupby(list(df.columns[:-1])).mean()
        model.fit(df.index.to_frame(), df['pconv'])
        # predict
        y_v = y.iloc[ival].to_numpy()
        p_pred = model.predict(X.iloc[ival])
        y_pred = (p_pred > thrs)
        # eval
        for i, j in np.ndindex((2, 2)):
            cms[i, j] += np.sum((y_v == i) * (y_pred == j), axis=-1)
    f1 = 2 * cms[1, 1] / (2*cms[1, 1] + cms[0, 1] + cms[1, 0])
    ithr = np.argmax(f1)
    return thrs[ithr, 0], cms[:, :, ithr]
    
    
class Classifier:
    """
    Classifier wrapper around a probability regressor with given decision
    threshold.
    """
    
    def __init__(self, model, dec_threshold: float)-> None:
        self.model = model
        self.thr = dec_threshold
        
    def predict(self, X: pd.DataFrame)-> np.ndarray:
        return (self.model.predict(X) > self.thr).astype(int)


# =============================================================================
#%% Data loading and initial preprocessing
# =============================================================================

# Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]

X_clf, y_clf = df.drop('converted', axis=1), df.loc[:, 'converted']

features = list(df.columns[:-1])
pconv = df.groupby(features).mean()
data = pd.DataFrame.from_dict(
    ({c: v.to_numpy() for c, v in pconv.index.to_frame().items()}
     | {'pconv': pconv['converted'].to_numpy()})
)

# data preparation
y = data.loc[:, 'pconv']
X = data.drop('pconv', axis=1)

# preprocessing
cat_vars = ['country', 'source']
bool_vars = ['new_user']
quant_vars = ['age', 'total_pages_visited']
col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
     ('bool_id', FunctionTransformer(None), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])
tree_col_preproc = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), cat_vars),
     ('id_', FunctionTransformer(None), bool_vars + quant_vars)])


# %% Gradient boosting regressor

# regressor
gbr = GradientBoostingRegressor(loss='squared_error',
                                n_estimators=1000,
                                max_depth=10)

# pipeline
pipeline = Pipeline([('column_preprocessing', tree_col_preproc),
                     ('regressor', gbr)])


gbr_model = GridSearchCV(
    pipeline,  param_grid={'regressor__max_depth': np.arange(2, 13, 1)},
    scoring='neg_mean_squared_error', n_jobs=4, refit=True, cv=10)
gbr_model.fit(X, y)
best_max_depth = gbr_model.best_params_['regressor__max_depth']
print(f'found best max_depth: {best_max_depth}')

pipeline['regressor'].max_depth = best_max_depth
thr, cm = tune_thr_cv(clone(pipeline), X_clf, y_clf)
print(f'best decision threshold: {thr}')
print_metrics(cm)

gbr_model = Classifier(gbr_model, thr)

"""
### evaluation
"""

# print('===== Gradient boosting regressor-baser classifier =====')
# _, cm = evaluate_model(gbr_model)
# print_metrics(cm)

sys.exit()


# %% SVM

# regressor
svr = SVR(kernel='rbf',
          gamma='scale',
          C=1,
          epsilon=0.1)

# pipeline
pipeline = Pipeline([('column_preprocessing', col_preproc),
                     ('regressor', svr)])

# parameter optimization
svr_model = GridSearchCV(
    pipeline,  param_grid={'regressor__C': np.logspace(-2, 1, 16)},
    scoring='neg_mean_squared_error', n_jobs=2, refit=True, cv=10)
svr_model.fit(X, y)
best_C = svr_model.best_params_['regressor__C']
print(f'found best C: {best_C}')

pipeline['regressor'].C = best_C
thr, cm = tune_thr_cv(clone(pipeline), X_clf, y_clf)
print(f'best decision threshold: {thr}')
print_metrics(cm)

svr_model = Classifier(svr_model, thr)


