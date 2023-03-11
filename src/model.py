"""This module contains machine learning model functionalities.

It creates, trains and evaluates the model.
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_and_evaluate_model(x_train, x_test, y_train, y_test):
    """Train and evaluate the model.
    
    Args:
        x_train: Training data.
        x_test: Test data.
        y_train: Training labels.
        y_test: Test labels.
    """

    dtrain = xgb.DMatrix(x_train, label=y_train)
    param = {'max_depth': 5, 'eta': 0.5, 'objective': 'multi:softmax', 'num_class': 4}
    param['nthread'] = 4
    param['eval_metric'] = 'mlogloss'
    evallist = [(dtrain, 'train')]
    num_round = 50
    bst = xgb.train(param, dtrain, num_round, evallist)

    dtest = xgb.DMatrix(x_test, label=y_test)
    ypred = bst.predict(dtest)

    print("Accuracy: ", accuracy_score(y_test, ypred))



