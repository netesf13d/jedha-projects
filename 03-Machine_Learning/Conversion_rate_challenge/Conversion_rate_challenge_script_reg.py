# -*- coding: utf-8 -*-
"""
# A regression approach to the classification problem

This script presents a different approach to the classification problem. Since we have many observations in a very low
dimension problem (only two quantitative features!), we can directly estimate the conversion probability density by regression.
The classification then amounts to determine if the estimated probability is above a certain threshold.
"""

import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler,
                                   FunctionTransformer)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


#%% Utils
"""
## <a name="utils"></a>Utilities

We introduce here relevant utilities that will make the code cleaner later on.
"""

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
        df = pd.concat([X.iloc[itr], y.iloc[itr]], axis=1)
        df = df.groupby(list(df.columns[:-1])).mean()
        model.fit(df.index.to_frame(), df['converted'])
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


#%% Data loading and initial preprocessing

"""
## <a name="loading"></a>Data loading and initial preprocessing

We prepare two datasets, one for plain classification, and one for conversion probability regression.
"""

# Load data
df = pd.read_csv('./conversion_data_train.csv')
df = df.loc[df['age'] < 80]

# prepare classification data
X_clf, y_clf = df.drop('converted', axis=1), df.loc[:, 'converted']

# prepare conversion probability data
features = list(df.columns[:-1])
pconv = df.groupby(features).mean()
data = pd.DataFrame.from_dict(
    ({c: v.to_numpy() for c, v in pconv.index.to_frame().items()}
     | {'pconv': pconv['converted'].to_numpy()})
)
y = data.loc[:, 'pconv']
X = data.drop('pconv', axis=1)

# columns preprocessings
cat_vars = ['country', 'source']
bool_vars = ['new_user']
quant_vars = ['age', 'total_pages_visited']


# %% Gradient boosting regressor

"""
## <a name="gbr"></a>Gradient boosting



### Model construction
"""

# column preprocessing
tree_col_preproc = ColumnTransformer(
    [('cat_oe', OrdinalEncoder(), cat_vars),
     ('id_', FunctionTransformer(None), bool_vars + quant_vars)])

# regressor
gbr = GradientBoostingRegressor(loss='squared_error',
                                learning_rate=0.1,
                                n_estimators=10000,
                                max_depth=2,
                                random_state=1234,
                                max_features=0.5)

# pipeline
pipeline = Pipeline([('column_preprocessing', tree_col_preproc),
                     ('regressor', gbr)])

# parameter optimization : `max_depth`
gbr_model = GridSearchCV(
    pipeline,
    param_grid={#'regressor__max_depth': np.arange(1, 4, 1),
                'regressor__learning_rate': np.logspace(-3, -1, 5)},
    scoring='neg_mean_squared_error', n_jobs=4, refit=True, cv=5)
gbr_model.fit(X, y)
best_learning_rate = gbr_model.best_params_['regressor__learning_rate']
print(f'found best learning_rate: {best_learning_rate}')
# best_max_depth = gbr_model.best_params_['regressor__max_depth']
# print(f'found best max_depth: {best_max_depth}')

pipeline['regressor'].learning_rate = best_learning_rate
# pipeline['regressor'].max_depth = 2 # best_max_depth
thr, cm = tune_thr_cv(clone(pipeline), X_clf, y_clf)
print(f'best decision threshold: {thr}')
print_metrics(cm)

"""

"""


"""
### Model evaluation on test data


"""

# print('===== Gradient boosting regressor-baser classifier =====')
# _, cm = evaluate_model(Classifier(gbr_model, thr))
# print_metrics(cm)
# sys.exit()



# %% SVM

"""
## <a name="svr"></a>Support vector machines

The reformulation of the classification problem into a regression task caused 20-times reduction in problem size.
This makes it affordable to use SVMs with non-linear kernels. 

In this section we fit a SVM with a radial-basis function kernel on the conversion probability. As above, we take advantage of the decreased
fitting times to optimize the model parameters `gamma` and `C`, along with the decision threshold. 

### Model construction
"""


# column preprocessing
col_preproc = ColumnTransformer(
    [('cat_ohe', OneHotEncoder(drop=None), cat_vars),
     ('bool_id', FunctionTransformer(None), bool_vars),
     ('quant_scaler', StandardScaler(), quant_vars)])

# regressor
svr = SVR(kernel='rbf',
          gamma='scale',
          C=1,
          epsilon=0.1)

# pipeline
pipeline = Pipeline([('column_preprocessing', col_preproc),
                     ('regressor', svr)])

"""
### Parameter optimization
"""

# parameter optimization
gamma_vals = np.repeat(np.logspace(-2, 1, 7), 7)
C_vals = np.tile(np.logspace(-2, 1, 7), 7)
epsilon_vals = np.logspace(-2, 1, 7)
svr_model = GridSearchCV(
    pipeline,
    param_grid={#'regressor__gamma': gamma_vals, 'regressor__C': C_vals,
                'regressor__epsilon': epsilon_vals},
    scoring='neg_mean_squared_error', n_jobs=6, refit=True, cv=5)
svr_model.fit(X, y)
# best_gamma = svr_model.best_params_['regressor__gamma'] # 0.1
# best_C = svr_model.best_params_['regressor__C'] # 1.
best_epsilon = svr_model.best_params_['regressor__epsilon']
# print(f'found best C: {best_C}; best gamma: {best_gamma}')
print(f'found best epsilon: {best_epsilon}')

# decision threshold tuning
# pipeline['regressor'].gamma = best_gamma
# pipeline['regressor'].C = best_C
pipeline['regressor'].epsilon = best_epsilon # 0.0316
thr, cm = tune_thr_cv(clone(pipeline), X_clf, y_clf)
print(f'best decision threshold: {thr}')
print_metrics(cm)

# TODO comment
"""
The cross-validation F1-score is 0.77, not better than what was obtained with a simple logistic regression with tuned decision threshold.
We note that the decision threshold maximizing the F1 score is around 0.40, close to the corresponding value of the tuned logistic regression.
This is a hint that the two models must behave very similarly.
"""


"""
### Model evaluation on test data
"""

# print('===== Gradient boosting regressor-baser classifier =====')
# _, cm = evaluate_model(Classifier(svr_model, thr))
# print_metrics(cm)

"""

"""

