"""
    Description:
        TreeTrainer class to handle training the models on data and
            storing the predictions run on test data

    ..notes:
        * only properly handles Scikit-Learn API

    TODO: insert a logger here

"""

# from functools import wraps

import numpy as np
import pandas as pd


class TreeTrainer(object):
    """ This object handles the training and """

    def __init__(self, models, report_dir, logger_on=False):
        """ Construct a TreeTrainer with a dictionary of models

        :param models: a dictionary mapping of names to models
        """
        if not isinstance(models, dict):
            raise ValueError("The passed parameter models should be a dictionary, it is type {} instead"
                             .format(type(models)))
        self._models = models
        self._report_dir = report_dir
        self._logger_on = logger_on
        self._y_preds = None  # this will be populated in the run_test method

    def fit_model(self, model_name, X_train, y_train, size_limit=None,
                  match_indices=False, **kwargs):
        """ Fit a single model (with model_name) to data y against X
        ..note: X_train and y_train should contain the dates of data

        :param model_name: (string) the model to fit
        :param X_train: the independent variables to train on
        :param y_train: the dependent variables to train on
        """
        if match_indices:
            if np.any(X_train.index != y_train.index):
                raise ValueError("The indices of X_test and y_test do not match")
        if size_limit is None:
            size_limit = X_train.shape[0]

        self._models[model_name].fit(X_train.tail(size_limit),
                                     y_train.tail(size_limit).values.ravel(), **kwargs)

    def fit_all(self, X_train, y_train, size_limit=None, match_indices=False, **kwargs):
        """ Fit all the models to data y against X
        ..note: X_train and y_train should contain the dates of data

        :param X_train: the independent variables
        :param y_train: the dependent variables
        """
        if match_indices:
            if np.any(X_train.index != y_train.index):
                raise ValueError("The indices of X_test and y_test do not match")
        if size_limit is None:
            size_limit = X_train.shape[0]

        for model_name in self._models.keys():
            self._models[model_name].fit(X_train.tail(size_limit),
                                         y_train.tail(size_limit).values.ravel(), **kwargs)

    def dump_test_predictions(self, file_path, append=False, wipe_data=False, **kwargs):
        """ This dumps the test predictions into a file

        :param file_path: the path to the file of where to dump the test results
        :param append: (boolean) if true, don't overwrite append to the file instead
            of overwriting
        :param wipe_data: (boolean) if true, wipe the data after dumping the test results
            to the file
        """
        if append:
            self._y_preds.to_csv(file_path, mode='a', header=False, **kwargs)
        else:
            self._y_preds.to_csv(file_path, mode='w', **kwargs)
        if wipe_data:  # wipe the data
            self._y_preds = None

    def get_test_predictions(self, model_name=None):
        """ Returns all the prediction values if called without arguments, otherwise
            return the prediction values only for a specific model
        """
        if model_name is None:
            return self._y_preds
        elif self._y_preds[model_name] is None:
            print("The predictions returned from model {} is None, should run predict class method \
                   to generate results".format(model_name))
        else:
            return self._y_preds

    def create_predictions(self, X_test, store_indices=False):
        """ Create model predictions for given set of data
        :param X_test: the independent variables of the test data
        :param store_indices: store the indices taken from X_test
        """
        y_preds_tmp = pd.DataFrame()  # temp. variable to append to class variable
        for model_name, model in self._models.items():
            y_pred = model.predict(X_test)
            y_preds_tmp[model_name] = y_pred

        if store_indices:
            y_preds_tmp.index = X_test.index
        if self._y_preds is None:
            self._y_preds = y_preds_tmp
        else:
            self._y_preds = self._y_preds.append(y_preds_tmp)
