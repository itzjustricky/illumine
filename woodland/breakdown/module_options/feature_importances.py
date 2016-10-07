"""
    Description:
        Use execution source to label which ones to execute?
        sort of hacky ...


    @author: Ricky Chang
"""

import numpy as np

from IPython.nbformat.v4.nbbase import (
    new_code_cell, new_markdown_cell, new_notebook,
    new_output, new_raw_cell
)


def generate_feature_importances(model_obj, n_features_to_display=10, n_features_per_cell=5):
    """ Generate a IPython Notebook cell that will produce the feature importances

    :param model_obj: python object of fitted Scikit-learn tree
    """
    cells = []
    cells.append(new_markdown_cell(source='# Feature Importance Analysis',))

    n_features_to_display = min(n_features_to_display, model_obj.feature_importances_.shape[0])
    n_features_per_cell = min(n_features_per_cell, n_features_to_display)

    # insert cell to import
    import_source = \
        """
            import numpy as np
            import matplotlib.pyplot as plt

            %matplotlib inline
        """
    cells.append(new_code_cell(
        source=import_source,
        execution_count=None,
    ))

    feature_clean_source = \
        """
            features_sorted = np.argsort(importances)[::-1]
            n_features_per_cell = 10
            n_splits = int(np.ceil(n_features / n_features_per_cell))
        """
    cells.append(new_code_cell(
        source=feature_clean_source,
        execution_count=None,
    ))

    n_splits = int(np.ceil(n_features_to_display / n_features_per_cell))
    for i in range(n_splits):
        feature_plot_source = \
            """
                # Plot the feature importances in a bar graph
                indexed_names = np.array(["{} ({})".format(name, i) for i, name in enumerate(names)])
                for i in range(n_splits):
                    tmp_inds = features_sorted[(i*n_features_per_cell):min((i+1)*n_features_per_cell, n_features)]
                    fig = plt.figure()
                    plt.title("Feature importances")
                    plt.bar(range(len(tmp_inds)), importances[tmp_inds], color="b")
                    plt.xticks(range(len(tmp_inds)), indexed_names[tmp_inds], rotation=30, fontsize=8)
                    plt.xlim([-1, len(tmp_inds)])
                    plt.tight_layout()
                    #fig.savefig("{}/feature_importance_barplot_run{}_{}.png".format(output_dir, i, period_stamp))
                    #fig.clear(); plt.close(fig)
            """
        cells.append(new_code_cell(
            source=feature_plot_source,
            execution_count=1,
        ))

    return cells


def generate_partial_dependence(tree_model):
    """
    """

    cells = []

    cells.append(new_markdown_cell(
        source='# Feature Importance Analysis',
    ))

    dependence_plot_source = \
        """
            n_features_per_cell = 6
            n_splits = int(np.ceil(n_features / n_features_per_cell))
            for i in range(n_splits):

                tmp_indices = list(range(i*n_features_per_cell, min((i+1)*n_features_per_cell, n_features)))
                if len(tmp_indices) <= 0: break
                fig, axs = plot_partial_dependence(regr, features_df, tmp_indices, feature_names=names,
                                                   n_jobs=3, grid_resolution=50, n_cols=3)
                fig.suptitle('Partial Dependence graph')
                plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
                fig.savefig("{}/one_way_dependence_run{}_{}.png".format(output_dir, i, period_stamp))
                fig.clear()
                plt.close(fig)

                # ------- Produce density plots of the year the model will be tested on as well ------- #
                fig = plt.figure()
                tmp_data = features_df[names[tmp_indices]]
                tmp_names = names[tmp_indices]
                tmp_data = tmp_data[(tmp_data.index >= start_date) & (tmp_data.index < end_date)]
                AX = gridspec.GridSpec(2, 3)
                AX.update(wspace=0.5, hspace=1)
                for idx, tup in enumerate(itertools.product(range(2), range(3))):
                    print(tup)
                    row, col = tup
                    tmp_name = tmp_names[idx]
                    ax_tmp = plt.subplot(AX[row, col])

                    sns.distplot(tmp_data[tmp_name], ax=ax_tmp)
                    xlim_min, xlim_max = ax_tmp.get_xlim()
                    plot_range = np.round(np.linspace(xlim_min, xlim_max, num=4), 2)
                    ax_tmp.set(xticks=plot_range)
                plt.axis('tight')
                fig.savefig("{}/one_way_density_run{}_{}.png".format(output_dir, i, period_stamp))
                fig.clear()
                plt.close(fig)
        """
    cells.append(new_code_cell(
        source=dependence_plot_source,
        execution_count=1,
    ))

    return cells
