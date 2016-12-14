"""
    Description:



    @author: Ricky Chang
"""

from math import ceil
import numpy as np

from ..woodland.leaf_analysis import get_tree_predictions


__all__ = ['feature_importance_barplot', 'active_leaves_boxplot',
           'step_improvement_plot']


def feature_importance_barplot(sk_ensemble, feature_names,
                               max_per_plot=10, n_ax_cols=3, bar_color='#A2F789',
                               n_features_to_display=None):
    """ Plot feature importances with horizontal bars

        All of the credit belongs to Peter Prettenhofer at Scikit-Learn
        License: BSD 3 clause

    """
    feature_importances = sk_ensemble.feature_importances_
    n_features = feature_importances.shape[0]

    if n_features_to_display is None:
        n_features_to_display = n_features

    # number of axes per row is determined by max_per_plot and n_ax_cols
    n_ax_rows = int(np.ceil(n_features_to_display / (max_per_plot * n_ax_cols)))

    # make importances relative to max importance
    feature_importances = 100.0 * (feature_importances / feature_importances.max())
    sorted_idx = np.argsort(feature_importances)[::-1][:n_features_to_display]
    pos = np.arange(max_per_plot) + .5

    import matplotlib.pyplot as plt
    fig, (all_axes) = plt.subplots(n_ax_rows, n_ax_cols)
    # this takes care of situation where n_ax_rows=1
    if not isinstance(all_axes, np.ndarray):
        all_axes = np.array([all_axes])

    for ax_ind, ax in enumerate(all_axes):
        start_ind = ax_ind * max_per_plot
        end_ind = min(start_ind + max_per_plot, n_features_to_display)
        feature_inds = sorted_idx[start_ind:end_ind][::-1]
        amount_to_fill = max_per_plot - len(feature_inds)

        features_to_plot = feature_importances[feature_inds]
        if amount_to_fill > 0:
            features_to_plot = \
                np.append(features_to_plot, np.zeros(amount_to_fill))

        ax.barh(pos, features_to_plot, align='center', color=bar_color)
        ax.set_yticks(pos)
        ax.set_yticklabels(feature_names[feature_inds])
        ax.set_xticks(np.linspace(0, 100, 6))
        ax.set_xlabel('Relative Importance')

    fig.tight_layout()
    return fig, all_axes


def step_improvement_plot(sk_ensemble, X, y, error_func=None):
    """ Plot the improvement per tree model added to the tree

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
        feature/dependent variables
    :param y: array_like or sparse matrix, shape = [n_samples, 1]
        target variable
    :param error_func: Defaults to None, if None it will take the error
        function from sk_ensemble. Calculates the error to be plotted.
        The function that takes arguments: pred, y

    """
    if not hasattr(sk_ensemble, 'staged_predict'):
        raise AttributeError("The sk_ensemble object passed does not have a staged_predict attribute")
    if error_func is None:
        error_func = sk_ensemble.loss_

    def stagewise_error(staged_preds, y_true, axis=1):
        errors = np.zeros(staged_preds.shape[0])
        for ind, col in enumerate(staged_preds):
            errors[ind] = error_func(col, y_true)
        return errors

    staged_preds = np.array(list(sk_ensemble.staged_predict(X)))
    staged_errors = stagewise_error(staged_preds, y)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(staged_errors)
    ax.set_xlabel('# of Estimators')
    ax.set_ylabel('Error')

    return fig, ax


def active_leaves_boxplot(sk_ensemble, X, n_ax_rows=1):
    """ Use a boxplot to plot the distribution

    :param sk_ensemble: scikit-learn ensemble model object
    :param X: array_like or sparse matrix, shape = [n_samples, n_features]
        feature/dependent variables
    """

    n_samples = X.shape[0]
    tree_predictions = get_tree_predictions(sk_ensemble, X)

    # Function to create a list of lists into boxplot format
    def make_boxplot_ready(np_array):
        formatted_data = [np_array[i, :] for i in range(np_array.shape[0])]
        return formatted_data

    formatted_predictions = make_boxplot_ready(tree_predictions)
    import matplotlib.pyplot as plt

    fig, (all_axes) = plt.subplots(n_ax_rows, 1)
    # this takes care of situation where n_ax_rows=1
    if not isinstance(all_axes, np.ndarray):
        all_axes = np.array([all_axes])

    datapts_per_ax = ceil(1.0 * n_samples / n_ax_rows)
    for ind, ax in enumerate(all_axes):
        subplt_slice = (datapts_per_ax * ind,
                        min(datapts_per_ax * (ind + 1), n_samples - 1))

        ax.boxplot(
            formatted_predictions[subplt_slice[0]:subplt_slice[1]],
            showfliers=False)
        ax.set_xticks(np.arange(1, datapts_per_ax + 1, datapts_per_ax // 10))
        ax.set_xticklabels(np.arange(1, datapts_per_ax + 1, datapts_per_ax // 10))
        ax.set_xlabel('Datarow #')
        ax.set_ylabel('Predictions')

    return fig, all_axes
