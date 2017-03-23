"""
    Module of Leaf objects which will store the
    data needed to decide the decision path and
    value at the terminal nodes.

"""

import numpy as np
cimport numpy as cnp


cdef bint le_relation(double value, double threshold) nogil:
    """ Less than or equal to relation function """
    return value <= threshold

cdef bint gt_relation(double value, double threshold) nogil:
    """ Strictly greater than relation function """
    return value > threshold


cdef class TreeSplit:

    def __cinit__(self, int feature, double threshold,
                  str feature_name, str relation, int print_precision):
        self.feature = feature
        self.threshold = threshold
        self.relation = relation
        self.feature_name = feature_name
        self.print_precision = print_precision
        self.float_formatter = '{{:.{}f}}'.format(print_precision)

        if relation == '<=':
            self.relate = le_relation
        elif relation == '>':
            self.relate = gt_relation
        else:
            raise ValueError("An invalid relation was passed, only "
                             "'<=' and '>' are supported")

    cdef bint apply(self, double value) nogil:
        return self.relate(value, self.threshold)

    def __str__(self):
        return "{{{}{}{}}}".format(
            self.feature_name, self.relation,
            self.float_formatter.format(self.threshold))

    def __repr__(self):
        return str(self)

    def __reduce__(self):
        return (self.__class__, (
            self.feature,
            self.threshold,
            self.feature_name,
            self.relation,
            self.print_precision)
        )


cdef class TreeLeaf:
    """ Object representation a leaf (terminal node) of a decision tree.
        Stores the path to the leaf and the value associated with the leaf.
    """

    def __cinit__(self, list tree_splits, double value):
        """
        :param tree_splits: the decision path to the node
        :param value: the value associated with the node
        """
        self.value = value
        self.tree_splits = tree_splits
        self.f_to_split_map = {}

        cdef set split_set = set()              # temp var to make hash id for leaf
        cdef int i, curr_feature, n_splits      # temp var to store value in loop
        cdef TreeSplit tree_split               # temp var to store value in loop
        n_splits = len(tree_splits)

        for i, tree_split in enumerate(tree_splits):
            split_set.add(str(tree_split))
            curr_feature = tree_split.feature
            if curr_feature not in self.f_to_split_map.keys():
                self.f_to_split_map[curr_feature] = []

            self.f_to_split_map[curr_feature].append(i)

        # store the hash id for the leaf
        self.leaf_hash = hash(frozenset(split_set))

        # convert all the lists in f_to_split_map into np.ndarrays
        # allows faster prediction since views can be used on np.ndarrays
        cdef int key
        cdef list v
        for key, v in self.f_to_split_map.items():
            self.f_to_split_map[key] = np.array(v, dtype=np.int32)

    cdef void apply(self, double[:, :] X, unsigned char[:] b_vector):
        """ Return a vector of size n (where n is # of rows/samples in X)

        :param X: 2d matrix of the NxK
        :param feature: the index of the feature
        """
        self._dense_apply(X, b_vector)

    cdef void _apply_to_feature(self, int feature, int[:] split_inds,
                                double[:, :] X, unsigned char[:] b_vector):
        """ This function iterates along the rows of X[:, feature]
            and does an & operation over all the tree splits to see if
            the value satisfies all relevant splits
        """
        cdef int n_samples = X.shape[0]
        cdef int n_splits = split_inds.shape[0]
        cdef int i, j
        cdef TreeSplit split

        # for each row of data, see if the row satisfies
        # all of the TreeSplits in the TreeLeaf
        for j in range(n_splits):
            split = self.tree_splits[split_inds[j]]
            with nogil:
                for i in range(n_samples):
                    b_vector[i] &= split.apply(X[i, feature])

    cdef void _dense_apply(self, double[:, :] X, unsigned char[:] b_vector):
        cdef int i, feature                         # ints used to iterate
        cdef cnp.ndarray[int, ndim=1] split_inds    # stores splits indices relevant to a feature

        # for all features in map see if relevant splits are satisfied
        for feature, split_inds in self.f_to_split_map.items():
            self._apply_to_feature(
                feature,
                split_inds,
                X, b_vector)

    def __str__(self):
        return '\n'.format(self.tree_splits)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.leaf_hash

    def __reduce__(self):
        return (self.__class__,
            self.tree_splits, self.value)
