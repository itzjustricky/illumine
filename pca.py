import bpdb
'''
    Description:
        Functions built to analyze features

    Notes:
        * strange ... eig decomposition doesn't match SVD decomposition ?...

    TODO:
        * function pca_breakdown maybe should return a matplotlib graph object instead

    @author: Ricky Chang
'''


import matplotlib.pyplot as plt
import numpy as np


def pca_breakdown(data_df, n_factors, mainx_labels=None, start=0, end=np.inf):
    """ From start to end, plot a heatmap for the PCA
        eigenvectors and for the dimensionally reduced factors

    :param data: the data stored in a dataframe indexed by dates
        with columns of factors
    :param n_factors: the number of PCA factors to include
    :param start: (optional) the starting index of which to do the analysis over
    :param end: (optional) the ending index of which to do the analysis over
    :returns: does not return anything; plots something
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    data_in_interval = data_df.loc[(data_df.index <= end) & (data_df.index >= start)]

    data_scaled = StandardScaler().fit_transform(data_in_interval)
    pca = PCA(n_components=n_factors)
    data_reduced = pca.fit_transform(data_scaled)

    S, V, D = np.linalg.svd(data_scaled)
    eig_vals, eig_vecs = V[:n_factors], D[:n_factors]
    feature_labels = np.array(data_in_interval.columns)
    feature_cnt = len(feature_labels)

    try:
        for vec in eig_vecs:
            np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0)
    except AssertionError:
        print("The norm-2 of eigen-vector does not compute to 1")
    try:
        np.testing.assert_almost_equal(np.dot(data_scaled, eig_vecs.T), data_reduced)
    except AssertionError:
        print("SVD decomposition factor reduction doesn't match scikit-learn pca fit_tranform.")

    import matplotlib.gridspec as gridspec
    AX = gridspec.GridSpec(2, 3)
    AX.update(wspace=0.5, hspace=1)
    plt.axis('tight')
    ax_eig = plt.subplot(AX[0, 0])
    ax_eigv = plt.subplot(AX[0, 1:])
    ax_main = plt.subplot(AX[1, :])

    # to see how the variance is distributed in the factors
    ax_eig.pcolor(eig_vals.reshape((len(eig_vals), 1)), cmap='YlGnBu')
    ax_eig.set_xticks([], minor=False)  # Get rid of tick labels for y-axis
    ax_eig.set_yticks(range(n_factors), minor=False)
    ax_eig.set_yticklabels(["fctr {}".format(i+1) for i in range(n_factors)], minor=False)
    ax_eig.set_title("PCA factor variance intensities")

    # see which feature produces the most variance
    ax_eigv.pcolor(eig_vecs, cmap='YlGnBu')
    ax_eigv.set_xticks(range(1, feature_cnt+1))
    ax_eigv.set_xticklabels(feature_labels, rotation=90)
    ax_eigv.set_yticks([], minor=False)  # Get rid of tick labels for y-axis
    ax_eigv.set_xlim([0, len(feature_labels)])
    ax_eigv.set_title("PCA eigen_vector intensities")

    # PCA factors intensities over the dataset
    ax_main.pcolor(data_reduced.T, cmap='YlGnBu')
    ax_main.set_xlim([0, data_reduced.shape[0]])
    ax_main.set_yticks(range(n_factors), minor=False)
    ax_main.set_yticklabels(["fctr {}".format(i+1) for i in range(n_factors)], minor=False)
    ax_main.set_title("The PCA Factor Intensities per datapoint")
    plt.show()


if __name__ == "__main__":
    print("Hello")
    bpdb.set_trace()  # ------------------------------ Breakpoint ------------------------------ #
