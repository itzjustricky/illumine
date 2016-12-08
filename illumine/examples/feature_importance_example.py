"""
    Description:



    @author: Ricky Chang
"""

import os
import inspect

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets.california_housing import fetch_california_housing

from illumine.woodland import (IPynbEnsembleManager, FeatureImportanceSnippet)


def main():

    # Set up the directory for the output
    file_dir = os.path.dirname(os.path.abspath(inspect.stack()[1][1]))
    output_dir = "{}/ft_tree_analysis".format(file_dir)

    cal_housing = fetch_california_housing()

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)
    names = cal_housing.feature_names
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)

    manager = IPynbEnsembleManager(clf, output_dir, 'clf_pickle.pkl')
    fi_snippet_kw = {'feature_names': names, 'features_to_display': [0, 3, 5, 6], 'run_flag': True}
    manager.add_snippet(FeatureImportanceSnippet(), **fi_snippet_kw)
    manager.save(notebook_name="fi_notebook.ipynb", version=4)


# Set main function for debugging if error
import bpdb, sys, traceback
if __name__ == "__main__":
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        bpdb.post_mortem(tb)
