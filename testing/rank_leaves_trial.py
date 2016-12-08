"""
    Description:
        Test the get_top_leaves method.


    @author: Ricky Chang
"""

import numpy as np

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle

from illumine import woodland


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

    trained_foliage = \
        woodland.aggregate_trained_leaves(clf, feature_names=boston.feature_names)
    activated_foliage = \
        woodland.aggregate_activated_leaves(clf, X_test, feature_names=boston.feature_names)

    n_unique_nodes = len(trained_foliage)
    top_trained_leaves = woodland.get_top_leaves(trained_foliage, rank='count', n_top=50)
    top_activated_leaves = woodland.get_top_leaves(activated_foliage, rank='count', n_top=len(trained_foliage))
    print("Top Trained Leaves")
    print(top_trained_leaves)
    print("Top Activated Leaves")
    print(top_activated_leaves)

    unique_leaves = woodland.unique_leaves_per_sample(clf, X_train, boston.feature_names)
    print(unique_leaves / n_unique_nodes)


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
