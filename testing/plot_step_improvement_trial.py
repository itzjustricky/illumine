"""
    Description:



    @author: Ricky Chang
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle

import illumine.visual


def main():

    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)

    illumine.visual.plot_step_improvement(clf, X_test, y_test)
    plt.show()


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
