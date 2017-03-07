"""
    Module of code to help tiling of data

    @author: Ricky Chang
"""

from collections import OrderedDict

import numpy as np


def create_data_tiles(n_bins, feature_names, data_matrix):
    """ Function to create DataTile objects over the
        passed; it is assumed that the each column in
        the data_matrix, is a vector of feature data
    """
    if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2:
        raise ValueError("Argument data_matrix should be a 2d numpy matrix.")
    if len(feature_names) != data_matrix.shape[1]:
        raise ValueError(
            "The # of columns in data_matrix should "
            "match the length of feature_names")

    data_tiles = dict()
    for feature_name, col in zip(feature_names, data_matrix.transpose()):
        data_tiles[feature_name] = DataTile(col, n_bins)

    return data_tiles


class DataTile(object):
    """ Object to represent the xxx-tile (e.g. quantile, decile)
        a data series. It is used so that you don't need to hold
        the whole data series in-memory to know the thresholds.
    """

    def __init__(self, data_series, n_bins):
        """
        :type data_series: array-like
        :type n_bins: int
        :param data_series: the data over which to get the tiling of
        :param n_bins: the # of bins of data, n_bins=10 will result
            in a decile
        """
        if data_series.ndim != 1:
            raise ValueError("The passed data_series argument should be 1-dimensional")
        if n_bins < 1:
            raise ValueError("Must pass in a value for n_bins of at least 1.")
        self._n_bins = n_bins

        self._thresholds = OrderedDict()
        ptiles = [(ind * 100 / n_bins) for ind in range(1, n_bins)]
        for ptile in ptiles:
            self._thresholds[ptile] = np.percentile(data_series, ptile)

        self._thresholds[0] = -np.inf
        self._thresholds[100] = np.inf

    @property
    def thresholds(self):
        return self._thresholds

    @property
    def n_bins(self):
        return self._n_bins

    def __str__(self):
        return "{}-tile".format(self.n_bins)

    def __repr__(self):
        return str(self)

    def find_tile_for(self, value):
        """ Do a binary search over the thresholds
            to find which bin the value belongs to

        :type value: float
        :param value: the value of which you want
            to find a bin for
        """
        l_bound, u_bound = 0, self.n_bins

        value_bin = self.n_bins // 2
        while u_bound != l_bound:
            if value <= self[value_bin]:
                u_bound = value_bin
            else:  # value > self[value_bin]:
                l_bound = value_bin + 1

            value_bin = (l_bound + u_bound) // 2

        return value_bin
