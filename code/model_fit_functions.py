import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from copy import copy, deepcopy
from sklearn.model_selection import ParameterGrid

from sklearn.preprocessing import StandardScaler


def standardise_data(x_train, x_val, x_test):
    scaler = StandardScaler()

    scaler.fit(x_train)

    train_df = pd.DataFrame(scaler.transform(x_train))
    val_df = pd.DataFrame(scaler.transform(x_val))
    test_df = pd.DataFrame(scaler.transform(x_test))

    return (train_df, val_df, test_df)


def fit_and_predict(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    y_predict_train = model.predict(x_train)
    y_predict_test = model.predict(x_test)

    return (y_predict_train, y_predict_test, model)


def r2_score_loss(y_true, y_pred):
    r = -r2_score(y_true, y_pred)
    return (r)


def model_tuning(model, par_grid, x_train, y_train, x_val, y_val, metric=r2_score):
    '''Fit and predict using a given models with a given parameter grid
    on train (fit & predict) and validation data (predict only).

    Inputs:
    - parameter grid: dictionary of parameters with attribute names as keys and values as items. For example,
    for a random forest model:

    rf_grid = {"n_estimators": [20, 40, 60],
          "max_depth": [5, 10, 15, 20]}

    - x_train: training set of features,
    - y_train: training set of labels
    - x_val: validation set of features
    - y_val: validation set of labels

    - metric: note that this needs to be in the form of loss, i.e. lower values are better.

    Returns:
    - dictionary of train and validation performance for different parameter values
    - best model as measured by validation set performance (object)
    - dictionary of predictions on train and val set for the top model'''

    # reshape response if needed
    if y_train.ndim > 1:
        y_train = np.array(y_train).reshape(-1)
    else:
        pass

    results = dict()  # dictionary to store results

    i = 0

    best_metric = +np.inf
    best_model = np.nan
    best_model_predictions = dict()

    # Loop through models defined in the grid
    for model_spec in list(ParameterGrid(par_grid)):
        print(model_spec)
        print("model iteration", i)
        i += 1

        model_instance = deepcopy(model)
        model_name = ""

        # set up attributes
        for attribute, value in model_spec.items():
            model_name = model_name + str(attribute) + ": " + str(value) + " "
            setattr(model_instance, attribute, value)  # set attribute values as specified in the grid

        print(model_instance)  # print to check

        # fit & predict
        y_predict_train, y_predict_val, model = fit_and_predict(model_instance, x_train, x_val, y_train)

        # metric results
        results[model_name] = dict()

        val_metric = metric(y_val, y_predict_val)

        results[model_name]["train_metric"] = np.round(metric(y_train, y_predict_train), 3)
        results[model_name]["val_metric"] = np.round(val_metric, 3)

        # keep track of the best model
        if val_metric < best_metric:
            print("New best metric:", val_metric)
            best_metric = val_metric
            best_model = model_instance
            best_model_predictions["y_predict_train"] = y_predict_train
            best_model_predictions["y_predict_val"] = y_predict_val

    return (results, best_model, best_model_predictions)


def model_tuning_cv(model, param_grid, x_train, y_train, x_test, score=None, cv=None):
    '''Fit and predict using cross-validation on train set.

    Inputs:
    - parameter grid: dictionary of parameters with attribute names as keys and values as items. For example,
    for a random forest model:

    rf_grid = {"n_estimators": [20, 40, 60],
          "max_depth": [5, 10, 15, 20]}

    - x_train: training set of features,
    - y_train: training set of labels
    - x_test: test set of features

    - score: one of Python's metric methods
    - cv: CV object

    Returns:
    - dictionary of CV results
    - best model as measured by CV
    - dictionary of predictions on test set'''

    # reshape response if needed
    if y_train.ndim > 1:
        y_train = np.array(y_train).reshape(-1)
    else:
        pass

    i = 0

    parameters = ParameterGrid(param_grid).__dict__["param_grid"][0]
    clf = GridSearchCV(model, parameters, cv=cv, scoring=score)
    clf.fit(x_train, y_train)

    best_model = clf.best_estimator_
    cv_results = clf.cv_results_

    best_model_predictions = dict()

    model_train = best_model.fit(x_train, y_train)
    best_model_predictions["y_test_predict"] = best_model.predict(x_test)

    return (cv_results, best_model, best_model_predictions)