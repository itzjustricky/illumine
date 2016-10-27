"""
    Description:
        Implementation of the FeatureImportanceSnippet to be outputted
        into an IPython notebook.

    TODO:
        * Change it so that the get_snippet function is done as a class method;
            should probably just combine the two classes below

    @author: Ricky Chang
"""

from copy import deepcopy
from collections import OrderedDict

from ...core import (ModelSnippet, format_snippet)


def format_pair(pair, format_index, *format_args):
    if len(pair) != 2:
        raise ValueError("The length > 2, this is not a pair.")

    formatted_item = pair[format_index].format(*format_args)

    if format_index == 0:
        return (formatted_item, pair[1])
    else:
        return (pair[0], formatted_item)


class FeatureImportanceTemplate(object):
    """ A wrapper around the snippet_dict to adhere to a more Pythonic coding style.
        The snippet_dict describes the template of which the Feature Importance
        snippet will take.
    """

    snippet_dict = OrderedDict()

    snippet_dict['snippet-title'] = ('markdown', "#{}")

    snippet_dict['import-cell'] = \
        ('code',
         """
            %matplotlib inline

            import math
            import pickle
            import numpy as np
            import matplotlib.pyplot as plt
         """)

    snippet_dict['load-cell'] = \
        ('code',
         """
            {0} = pickle.load(open('pickles/{1}', 'rb'))
            all_feature_importances = {0}.feature_importances_
            params = {0}.get_params()

            feature_names = np.array({2})
         """)

    snippet_dict['adjust-cell'] = \
        ('code',
         """
             # Parameters passed into function here
             features_to_display = {}
             n_features_to_display = len(features_to_display)
             max_per_plot = {}

             # limit to 10 features per plot and 3 plots per row
             n_ax_cols = min(3, math.ceil(n_features_to_display / max_per_plot))
             n_ax_rows = math.ceil(n_features_to_display / max_per_plot / n_ax_cols)
         """)

    snippet_dict['plots-title'] = \
        ('markdown',
         """
             <font color="blue">
             ## {}
         """)

    snippet_dict['plot-cell'] = \
        ('code',
         """
             # make importances relative to max importance
             feature_importance = 100.0 * (all_feature_importances / all_feature_importances.max())
             sorted_idx = np.argsort(feature_importance)[::-1]
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
                     features_to_plot = np.append(features_to_plot, np.zeros(amount_to_fill))


                 ax.barh(pos, features_to_plot, align='center', color='#A2F789')
                 ax.set_yticks(pos)
                 ax.set_yticklabels(feature_names[feature_inds])
                 ax.set_xticks(np.linspace(0, 100, 6))
                 ax.set_xlabel('Relative Importance')

             fig.tight_layout()
             plt.show()
         """)

    @classmethod
    def get_template(cls):
        """ This method should be the only means of accessing snippet_dict """
        snip = deepcopy(cls.snippet_dict)
        return snip


class FeatureImportanceSnippet(ModelSnippet):
    """ Object to handle the preparation of the snippet for FeatureImportance """

    @classmethod
    def display_info(cls):
        """ Print out the parameters needed for the generate_snippet function """
        print(cls.generate_snippet.__doc__)

    def generate_snippet(self, feature_names, features_to_display,
                         model_id='sk_model', max_per_plot=10, run_flag=False,
                         snippetmd_title='Feature Importance Analysis',
                         plotsmd_title='Bar Plots'):
        """
        Details about the parameters needed for generating a FeatureImportanceSnippet:

        Required parameters
        ===================
        :param pickle_file (string): the string of the name of the pickle file
        :param feature_names: list-type object containing strings of the names of the features.
            If it is not a numpy array, it will be transformed to one;
            it needs the numpy array index selection feature
        :param features_to_display: list-type object containing the indices of the features to display

        Parameters with Default Values
        ==============================
        :param model_id (string): (default='sk_model') the identifier used for the Sklearn model
        :param max_per_plot (int): (default=10) the number of bars to graph per plot
        :param run_flag (bool): (default=False) if True, run all the code cells in the snippet;
            NOTE it will take LONGER for program to run
        :param snippetmd_title (string): (default='Feature Importance Analysis') the title to be
            placed at the top of the snippet as a markdown cell; if None then no cell will be inserted
        :param plotsmd_title (string): (default='Bar Plots') the title to be placed before the cell plot;
            if None then no cell will be inserted
        """
        pickle_file = self._pickle_file
        fi_template = FeatureImportanceTemplate.get_template()

        # Delete the markdown cells if no title is provided
        if snippetmd_title is None:
            fi_template.pop('snippet-title')
        else:
            fi_template['snippet-title'] = format_pair(fi_template['snippet-title'], 1,
                                                       snippetmd_title)
        if plotsmd_title is None:
            fi_template.pop('plots-title')
        else:
            fi_template['plots-title'] = format_pair(fi_template['plots-title'], 1,
                                                     plotsmd_title)

        fi_template['load-cell'] = format_pair(fi_template['load-cell'], 1, model_id,
                                               pickle_file, feature_names.__str__().replace('\n', ''))
        fi_template['adjust-cell'] = format_pair(fi_template['adjust-cell'], 1,
                                                 features_to_display, max_per_plot)

        return format_snippet(list(fi_template.values()), run_flag)