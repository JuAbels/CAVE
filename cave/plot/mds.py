"""
Multi-dimensional Scaling (MDS) with SMACOF Algorithm.

Done as Undergraduate Thesis. Adopted by scikit-learn.
"""

import numpy as np
from numpy import linalg as la
import warnings
from math import acos, degrees, cos, sin
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.manifold.mds import smacof


def calculate_angle(random_configuration):
    """
    Helpfunction to calculate the angel between new point, which has to be
    inserted in 2D, and all configurations which are in 2d already.

    Parameters
    ----------
    random_configuration: np.array, shape (n_confis, n_featues)
        Matrix with configuration in original space.

    Returns
    -------
    angle: np.arry, shape(n x 1)
        Vector with all angles in dregees of the configuration which has to be
        inserted and each random configuration.
    """
    actual_confi = random_configuration[-1]
    angle = []

    for other_confis in range(random_configuration.shape[0] - 1):
        theta = degrees(acos(np.dot(random_configuration[other_confis], actual_confi) /
                             (np.sqrt(np.dot(random_configuration[other_confis], random_configuration[other_confis])) *
                              np.sqrt(np.dot(actual_confi, actual_confi)))))
        angle.append(theta)

    angle = np.array(angle)

    return angle


def calculate_new_points(embedding, distances, angle):
    """
    Calculate position of new point in lower dimensional. Thereby calulate with formula

        x_1 = x_2 + distance * cos(theta)
        x_y = x_y + distance * cos(theta)

    Caculate first all x and y pairs of the new point and take then average of all x and y.

    Parameters
    ----------
    embedding: np.array, shape(n_configurations, lower dimesion)
        matrix of acutal new space with present configurations
    distances: np.array, shape(n_configurations, n_configurations)
        vector of actual distances of configurations in 2D space between new point
        and all other other configurations
    angle: np.array, shape(n_configurations, 1)
        vector with all angels between new point and all present configurations

    Returns
    -------
    x: float
        average float of all x positions
    y: float
        average float of all y positions
    """
    assert(embedding.shape[0] == distances.shape[0] == angle.shape[0])
    x_list = []
    y_list = []
    for counter in range(embedding.shape[0]):
        # calculate x
        x_list.append(embedding[counter][0] + distances[counter] * cos(angle[counter]))
        y_list.append(embedding[counter][1] + distances[counter] * sin(angle[counter]))
    return sum(x_list)/len(x_list), sum(y_list)/len(y_list)


def get_distances(matrix, depth, is_cat):
    """
    Helpfunction.
    """
    n_confs = matrix.shape[0]
    dists = np.zeros((n_confs, n_confs))

    for i in range(n_confs):
        for j in range(i + 1, n_confs):
            dist = np.abs(matrix[i, :] - matrix[j, :])
            dist[np.isnan(dist)] = 1
            dist[np.logical_and(is_cat, dist != 0)] = 1
            dist /= depth
            dists[i, j] = np.sum(dist)
            dists[j, i] = np.sum(dist)

    return dists


class MDS_New(BaseEstimator):
    """Multidimensional scaling

    Read more in the :ref:`User Guide <multidimensional_scaling>`.

    Parameters
    ----------
    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities.

    metric : boolean, optional, default: True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.

    n_init : int, optional, default: 4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    method : string, optional, default: 'smacof'
            By default, the smacof algorithm is used and the model can not
            transform new data points. With method='inductive' the model is built
            using eigen decomposition, allowing for the transform method to be
            used with new data.


    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    "Out-of-sample extensions for lle, isomap, mds, eigenmaps, and spectral
    clustering." Bengio, Y. et al. Advances in neural information processing
    systems 16 (2004): 177-184.

    """
    def __init__(self, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean",
                 method="smacof"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.method = method

    def test_Case(self):
        test_MDS_transform()

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None):
        """
        Computes the position of the points in the embedding space

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored

        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        # self.fit_transform(X, init=init)
        if self.method == 'inductive':
            self.X_train_ = X
            if self.dissimilarity == 'precomputed':  # call of this case, because is precomputed
                D = X                                # X is the distance matrix of random configurations
            elif self.dissimilarity == 'euclidean':
                D = euclidean_distances(X)
                self.D_XX_ = euclidean_distances(self.X_train_, self.X_train_)
            else:
                raise ValueError("Proximity must be 'precomputed' or"
                                 "'euclidean'."
                                 " Got %s instead" % str(self.dissimilarity))

            # Normalising similarities
            n = len(D)  # dimension of distance matrix
            D_sq = np.square(D)  # squared of distance matrix
            P = np.eye(n) - 1 / n * np.ones((n, n))  # centering matrix, mit I_N - 1/N * O = P
            K = -0.5 * np.dot(np.dot(P, D_sq), P)    # double centering  - 1/2 (H*D*H), normalized Matrix

            # Sorting e-vectors and e-values according to e-val
            e_vals, e_vecs = la.eigh(K)
            ind_sort = np.argsort(e_vals)[::-1]
            self.e_vecs_ = e_vecs[:, ind_sort]
            self.e_vals_ = e_vals[ind_sort]
        else:
            self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X_train, X_New=None, init=None):
        """Fit the data from X, and returns the embedded coordinates

        Parameters
        ----------
        X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                if dissimilarity='precomputed'
        init : ndarray, shape (n_samples,), optional, default: None
            Should only be used with the non-inductive MDS (SMACOF).
            If None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array.
        """
        # Fit the data from X, and returns the embedded coordinates
        if self.method == 'inductive':
            if init is not None:
                raise ValueError("Init is only for the non-inductive MDS.")
            ret = self._fit_transform_ext(X_train, X_New)
        else:
            ret = self._fit_transform(X_train, init)
        return ret

    def transform(self, X):
        """
        Apply the transformation on X
        If dissimilarity is Euclidean, apply the transformation on X.
        If dissimilarity is precomputed, X is the similarity matrix to be used
        between new (out-of-sample) points with old ones.
        The new points (X if Euclidean, or with X similarity matrix if
        precomputed) are projected in the same space as the training set.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        X : array, shape [n_samples, n_features], or \
                [n_samples, n_train_samples] if dissimilarity='precomputed'
            New data, where n_samples is the number of samples
            and n_features is the number of features for "euclidean"
            dissimilarity. Else, similarity matrix (e.g. Euclidean distances
            between new and training points).
        y: Ignored
            NB: similarity matrix has to be centered, use the
            make_euclidean_similarities function to create it.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        if not self.method == 'inductive':
            raise ValueError("Method only available if inductive is True.")
        if self.dissimilarity == 'precomputed':
            D_new = X
        elif self.dissimilarity == 'euclidean':
            if not hasattr(self, 'X_train_') \
                    or not hasattr(self, 'D_XX_') \
                    or self.X_train_ is None \
                    or self.D_XX_ is None:
                raise ValueError("Inductive MDS with Euclidean Similarities "
                                 "must be fit first. use ``MDS.fit()``")
            else:
                D_aX = euclidean_distances(X, self.X_train_)
                D_new = self.center_similarities(D_aX, self.D_XX_)
        else:
            raise ValueError("Dissimilarity not set properly: 'precomputed' "
                             "and 'euclidean' allowed.")
        X_new = self._mds_project(D_new, k=self.n_components)
        return X_new

    def _fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored

        init : ndarray, shape (n_samples,), optional, default: None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state,
            return_n_iter=True)

        return self.embedding_

    def _fit_transform_ext(self, X_train, X_new):
        # Make fitting with random distances
        self.fit(X_train)

        # Transform the random configurations to two dimensions
        training_points = self.transform(X_train)

        # Calculate center similarities
        new_traing_dists = self.center_similarities(X_new, X_train)

        # Transform the new points
        new_points = self.transform(new_traing_dists)

        dists = np.vstack((training_points, new_points))
        return dists

    def center_similarities(self, D_aX, D_XX):
        """
        Centers similarities D_aX around D_XX

        Parameters
        ----------
        D_aX : array, shape=[n_new_samples, n_train_samples]
            Dissimilarity matrix of new and training data.
        D_XX : array, shape=[n_train_samples, n_train_samples]
            Dissimilarity matrix of training data.

        Returns
        -------
        new_similarities : array-like, shape=[n_new_samples, n_train_samples]
        """
        D_aX = np.square(D_aX)
        D_XX = np.square(D_XX)
        N = len(D_XX)
        M = len(D_aX)
        I_NN = np.ones((N, N))
        I_MN = np.ones((M, N))
        Exp_XX = np.sum(D_XX) / N ** 2

        new_similarities = -0.5 * (D_aX - (np.dot(D_aX, I_NN) +
                                           np.dot(I_MN, D_XX)
                                           ) / N + Exp_XX)
        return new_similarities

    def _mds_project(self, new_similarities, k):
        """

        Parameters
        ----------
        new_similarities:
        k: Number of dimensions of the embedding space.

        Returns
        -------
        e_projections:
        """

        e_projections = np.zeros((len(new_similarities), k))

        for i in range(len(new_similarities)):
            for j in range(k):
                e_projections[i, j] = ((np.dot(self.e_vecs_[:, j],
                                               new_similarities[i]) /
                                        np.sqrt(self.e_vals_[j])))
        e_projections = np.dot(new_similarities,
                               np.dot(self.e_vecs_[:, :k],
                                      np.diag(1/np.sqrt(self.e_vals_[:k]))))
        return e_projections




    def transform_wrong(self, X, random_config, confi_matrix, rhs, depth, is_cat):
        """
        TODO there are ways to extend MDS to provide a transform-method. if available,
            train on randomly sampled configs and plot all

        Works with the fit method to learn how to transform points in embedding space.
        Get through all configurations and depict new point with same euclidean distance.

        First get all distances from fitted random space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data of all datapoints.

        random_config: np.array, shape (n_config, m_dimensions)
            Configuration of all random configurations in high-dimensional. Extend this Matrix, so that
            in the end all configurations are in this matrix -> random_config.shape == confi_matrix.shape

        Returns
        -------
        embedded : np.array
            Matrix of all coordinates of the configurations in embedded space.

        TODO NaN is problematic -> eliminiate in for angle
            Problem: Categorical hyperparamters. Tranform to value 0 < x < 1 -> dann muss
                     transformation von random_configuration durchgeführt werden & und von confi
                     jeweils im run
        """
        random_config[np.isnan(random_config)] = 0  # delete all nan's so that calculations are possible

        for confi in rhs.config_ids:
            confi = confi.get_array()
            confi[np.isnan(confi)] = 0

            # calculate the distances
            random_config = np.vstack((random_config, np.array([confi])))
            # dist_high_space = np.array(get_distances(random_config, depth, is_cat)[-1][1:])
            dist_high_space = np.array(euclidean_distances(random_config)[-1][1:])

            # Helpfunction which calculate x and y in 2D
            angle = calculate_angle(random_config)
            x, y = calculate_new_points(self.embedding_, dist_high_space, angle)

            self.embedding_ = np.vstack((self.embedding_, np.array([x, y])))

            test1 = np.array(euclidean_distances(self.embedding_)[-1])
            test2 = np.array(euclidean_distances(random_config)[-1])
            stress = (test1 - test2)

            print("Funktion fertig")

        assert(random_config.shape == confi_matrix.shape)
        assert(random_config.shape[0] == self.embedding_.shape[0])
        return self.embedding_


def test_MDS_transform():
    sim = np.array([[0, 5, 3, 4],
                    [5, 0, 2, 2],
                    [3, 2, 0, 1],
                    [4, 2, 1, 0]])

    mds_clf = MDS_New(dissimilarity="euclidean", method="inductive")
    mds_clf.fit(sim)

    # Testing for extending MDS to new points
    sim2 = np.array([[3, 1, 1, 2],
                     [4, 1, 2, 2]])
    result = mds_clf.transform(sim2)
    expected = np.array([[-.705, -.452],
                         [-1.051, -1.329]])
    print(result)
    np.testing.assert_array_almost_equal(result, expected, decimal=3)

