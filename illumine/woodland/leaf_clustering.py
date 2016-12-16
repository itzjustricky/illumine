"""
    Module to contain functions for leaf clustering

"""

import numpy as np

from .optimized_predict import map_features_to_int
from .optimized_predict import find_activated


def compute_activation(lucid_ensemble, X_df, considered_leaves=None):
    """ Compute an activation matrix to be used as vectors for
        clustering leaves together
    """

    f_map = map_features_to_int(X_df.columns)
    X = X_df.values

    considered_leaf_strings = [' & '.join(leaf) for leaf in considered_leaves]
    if considered_leaves is not None:
        filtered_leaves = \
            dict([(key, val) for key, val in lucid_ensemble.compressed_ensemble.items()
                  if key in considered_leaf_strings])
    else:
        filtered_leaves = lucid_ensemble.compressed_ensemble.items()

    activation_matrix = np.zeros(
        (lucid_ensemble.unique_leaves_per_sample, X_df.shape[0]),
        dtype=bool)

    for ind, leaf_pair in enumerate(filtered_leaves):
        path, value = leaf_pair
        activated_indices = find_activated(X, f_map, path.split(' & '))
        activation_matrix[ind, np.where(activated_indices)[0]] = 1

    return activation_matrix


def kmeans_leaves_cluster(n_clusters, lucid_ensemble, X_df, criterion):

    pass


if __name__ == '__main__':
    pass
