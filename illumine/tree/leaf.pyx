"""
    Module of Leaf objects which will store the
    data needed to decide the decision path and
    value at the terminal nodes.

"""

import numpy as np
cimport numpy as cnp


cdef class TreeSplit:
    """ Representation of TreeSplit which contains
        feature_name, relation, threshold of a
        decision tree split
    """

    def __cinit__(self,
                  int feature,
                  str feature_name,
                  str relation,
                  double threshold,
                  int print_precision):
        self.feature = feature
        self.feature_name = feature_name
        self.relation = relation
        self.threshold = threshold
        self.print_precision = print_precision

    cdef bint apply(self, double value):
        if self.relation == '<=':
            return value <= self.threshold
        else:
            return value > self.threshold

    def __str__(self):
        return "TS({}{}{})".format(
            self.feature_name,
            self.relation,
            format(self.threshold,
                   '.{}f'.format(self.print_precision))
        )

    def __reduce__(self):
        return (self.__class__, (
            self.feature,
            self.feature_name,
            self.relation,
            self.threshold)
        )

    def __repr__(self):
        return str(self)

    def __richcmp__(x, y, int op):
        if op == 0:
            return str(x) < str(y)
        if op == 2:
            return str(x) == str(y)
        if op == 4:
            return str(x) > str(y)
        if op == 1:
            return str(x) <= str(y)
        if op == 3:
            return str(x) != str(y)
        if op == 5:
            return str(x) >= str(y)


cdef class DecisionPath:
    """ Object representation of the path to a leaf node """

    def __cinit__(self, list tree_splits):
        """ The initializer for LeafPath

        :param tree_splits (list): a list of TreeSplit objects which
            is defined in the Cython module leaf_retrieval

            A TreeSplit represent a single split in feature data X.
        """
        # used to map features to relevant TreeSplits
        self.feature_map = {}
        cdef str key_name_part, key_rel_part
        key_name_part = ''
        key_rel_part = ''

        cdef int i, curr_feature
        # loop through TreeSplits, construct features_mapper & key
        for i in range(len(tree_splits)):
            curr_feature = tree_splits[i].feature

            # construct feature_mapper
            if curr_feature in self.feature_map:
                self.feature_map[curr_feature].append(tree_splits[i])
            else:
                self.feature_map[curr_feature] = [tree_splits[i]]

            # construct key
            key_name_part += tree_splits[i].feature_name
            key_rel_part += tree_splits[i].relation
            key_rel_part += format(
                tree_splits[i].threshold,
                '.{}f'.format(tree_splits[i].print_precision)
            )

        # the key is ordered so that name comes before the relation so
        # TreeSplits with the same features are grouped together for printing
        self.key = key_name_part + key_rel_part
        self.relevant_features = \
            np.array(list(self.feature_map.keys())) \
            .astype(dtype=np.int32)

    # I don't think this function should be here
    cdef bint apply(self, double value, int feature):
        """ Return true if the argument value for the passed
            feature satisfies all the TreeSplits
        """
        cdef TreeSplit treesplit_tmp

        if feature in self.feature_map:
            for treesplit_tmp in self.feature_map[feature]:
                if not treesplit_tmp.apply(value):
                    return False
            return True
        else:
            raise ValueError(
                "Feature with index {} is not part of this DecisionPath"
                .format(feature))

    def __iter__(self):
        return iter(self.feature_map.values())

    def __str__(self):
        return str(list(self.feature_map.values()))

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.tree_splits)

    def __hash__(self):
        return hash(self.key)

    def __richcmp__(self, other, int op):
        if op == 0:
            return self.key < other.key
        if op == 2:
            return self.key == other.key
        if op == 4:
            return self.key > other.key
        if op == 1:
            return self.key <= other.key
        if op == 3:
            return self.key != other.key
        if op == 5:
            return self.key >= other.key


cdef class TreeLeaf:
    """ Object representation a leaf (terminal node) of a decision tree.
        Stores the path to the leaf and the value associated with the leaf.
    """

    def __cinit__(self, list tree_splits, double value, int n_samples):
        """
        :type tree_splits: list of TreeSplit objects
        :param tree_splits: the decision path to the node
        :param value: the value associated with the node
        :param n_samples: the number of samples that reach the node
        """
        self.decision_path = DecisionPath(tree_splits)
        self.value = value
        self.n_samples = n_samples

        self.relevant_features = self.decision_path.relevant_features

    def __str__(self):
        cdef list node_strings = []
        cdef list keys, values

        if self._cached_repr is None:
            keys = ["path", "value", "n_samples"]
            values = [self.decision_path, self.value, self.n_samples]

            for key, val in zip(keys, values):
                node_strings.append("{}: {}".format(key, val))
            self._cached_repr = "({})".format(', '.join(node_strings))

        return self._cached_repr

    def __repr__(self):
        return str(self)

    def apply(self, double[:, :] X, int feature=-1):
        """ Return a vector of size n (where n is the # of rows/samples in X)

        :param X: 2d matrix of the NxK
        :param feature: the index of the feature
        """
        return self._dense_apply(X, feature)

    cdef int[:] _dense_apply(self, double[:, :] X, int feature):
        cdef int i, k, curr_feature
        cdef int N = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef int[:] res = np.ones(N, dtype=np.int32)

        if feature == -1:
            for i in range(N):
                for k in range(len(self.relevant_features)):
                    curr_feature = self.relevant_features[k]
                    res[i] &= self.decision_path.apply(
                        X[i, curr_feature], curr_feature)

        else:   # a feature was passed
            for i in range(N):
                res[i] = self.decision_path.apply(
                    X[i, feature], feature)

        return res
