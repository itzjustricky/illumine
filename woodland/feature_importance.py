"""
    Description:
        Use execution source to label which ones to execute?
        sort of hacky ...

    TODO:
        * think about whether or not to use a global variable
            and if use, at least think of a better way

    @author: Ricky Chang
"""

from collections import OrderedDict


global fi_snippet = OrderedDict()

fi_snippet['markdown-title'] = "{}"

fi_snippet['import-cell'] = \
    """
        import math
        import pickle
        import numpy as np
        import matplotlib.pyplot as plt
    """


fi_snippet['load-cell'] = \
    """
        clf = pickle.load(open('boston_fitted_gbregr.pkl', 'rb'))
        params = clf.get_params()
        feature_names = {}
    """

fi_snippet['adjust-cell'] = \
    """
        # Parameters passed into function here
        n_features_to_display = {}
        max_per_plot = {}

        # limit to 10 features per plot and 3 plots per row
        n_ax_cols = min(3, math.ceil(n_features_to_display / max_per_plot))
        n_ax_rows = math.ceil(n_features_to_display / max_per_plot / n_ax_cols)
    """

fi_snippet['before-plots-title'] = \
    """
        <font color="blue">
        ## Feature Importance Bar Plots
    """

fi_snippet['plot-cell'] = \
    """
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)[::-1][:n_features_to_display]
        pos = np.arange(max_per_plot) + .5
        fig, (all_axes) = plt.subplots(n_ax_rows, n_ax_cols)

        if not isinstance(all_axes, np.ndarray):
            all_axes = np.array([all_axes])

        for ax_ind, ax in enumerate(all_axes):
            start_ind = ax_ind * max_per_plot
            end_ind = min(start_ind + max_per_plot, n_features_to_display)
            feature_inds = sorted_idx[start_ind:end_ind][::-1]
            amount_to_fill = max_per_plot - len(feature_inds)

            features_to_plot = feature_importance[feature_inds]
            if amount_to_fill > 0:
                features_to_plot = \
                    np.append(features_to_plot, np.zeros(amount_to_fill))


            ax.barh(pos, features_to_plot, align='center')
            ax.set_yticks(pos)
            ax.set_yticklabels(feature_names[feature_inds])
            ax.set_xticks(np.linspace(0, 100, 6))
            ax.set_xlabel('Relative Importance')

        fig.tight_layout()
        plt.show()
    """

class FeatureImportanceSnippet(object):
    """ Object to represent the snippet of FeatureImportance
    """

    def __init__(self, pickle_file):
        """

        :param pickle_path (string): the string to the path of the pickle file
        """
        self._cells = []

        pass
