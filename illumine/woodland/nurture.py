"""
    This module contains methods for a set
    of algorithms to iteratively prune ensembles
    and build more ensembles on top

"""

import logging
from .ensemble_factory import make_LucidEnsemble

from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


# TODO: docstring for weighted_nurturing
def weighted_nurturing(ensemble_class, X_train, y_train,
                       feature_names, n_iterations, n_tunes=50, chunk_size=100,
                       metric_function='mse', update_weight=0.5, zero_bounds=0.001,
                       print_precision=5, tree_kws=None, l2_coef=0.0,
                       model_params=None, ensemble_kws=None):
    """ Fitting of a tree ensemble through the weighted scheme

    :param ensemble_class: the class for the ensemble model
    """
    if model_params is None:
        model_params = dict()
    if ensemble_kws is None:
        ensemble_kws = dict()

    self_weight = 1 - update_weight
    nurtured_model = None
    ensemble_model = ensemble_class(**model_params)
    for ind in range(n_iterations):
        if nurtured_model is None:
            new_y_train = y_train
        else:
            new_y_train = y_train - self_weight * \
                nurtured_model.predict(X_train)

        ensemble_model.fit(X_train, new_y_train)
        compressed_ensemble = make_LucidEnsemble(
            ensemble_model, feature_names,
            print_precision=print_precision,
            tree_kws=tree_kws, ensemble_kws=ensemble_kws).compress()

        if nurtured_model is None:
            nurtured_model = compressed_ensemble
        else:
            nurtured_model.combine_with(
                compressed_ensemble,
                self_weight=self_weight,
                other_weight=1.0)

        # compressed_ensemble.prune_by_leaves(
        #     X_train, y_train,
        #     metric_function=metric_function,
        #     n_tunes=n_tunes)
        compressed_ensemble.finetune(
            X_train, y_train, n_tunes, chunk_size=chunk_size,
            metric_name=metric_function, zero_bounds=zero_bounds, l2_coef=l2_coef)
        print("The error is {} on iteration {}".format(
            mean_squared_error(compressed_ensemble.predict(X_train), y_train)))

    return nurtured_model


# TODO
def boosted_nurturing():
    """ Fitting of a tree ensemble through the weighted scheme """
    pass


if __name__ == '__main__':
    pass
