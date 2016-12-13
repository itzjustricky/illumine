"""
    Description:
        Contains functions to generate rank methods for
        use in methods in rank_leaves & rank_leaves_per_point
        in the woodland/leaf_analysis.py module


    @author: Ricky Chang
"""


def generate_error_rank(true_y, error_method):
    """ Generate negative-error function so that
        error creates
    :param true_y: a vector of the true y-values that a
        model is trying to predict that is accepted by
        the error_method passed in
    :param error_method (function): a function that takes two
        vector-like parameters and outputs some error measure
    """

    def negative_error(X):
        return -error_method(X, true_y)

    return negative_error


if __name__ == '__main__':
    pass
