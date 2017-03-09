"""
    This module contains methods for a set
    of algorithms to iteratively prune ensembles
    and build more ensembles on top

"""

from .ensemble_factory import make_LucidEnsemble


# TODO: docstring for weighted_nurturing
def weighted_nurturing(ensemble_class, X_train, y_train,
                       feature_names, n_iterations, n_prunes=None,
                       metric_function='mse', update_weight=0.5,
                       print_precision=5, tree_kws=None,
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

        compressed_ensemble.prune_by_leaves(
            X_train, y_train,
            metric_function=metric_function,
            n_prunes=n_prunes)

    return nurtured_model


# TODO
def boosted_nurturing():
    """ Fitting of a tree ensemble through the weighted scheme """
    pass


if __name__ == '__main__':
    pass
