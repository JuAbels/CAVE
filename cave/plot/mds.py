"""
Multi-dimensional Scaling (MDS) with SMACOF Algorithm.

Done as Undergraduate Thesis. Adopted by scikit-learn.
"""

import numpy as np
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

    """
    def __init__(self, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

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
        self.fit_transform(X, init=init)
        return self

    def transform(self, X, random_config, confi_matrix, rhs, init=None):
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
        """
        for confi in rhs.config_ids:
            confi = confi.get_array()

            # calculate the distances
            distances = euclidean_distances(self.embedding_)
            random_config = np.vstack((random_config, np.array([confi])))
            dist_high_space = np.array(euclidean_distances(random_config)[-1][1:])

            # Helpfunction which calculate x and y in 2D
            angle = calculate_angle(random_config)
            x, y = calculate_new_points(self.embedding_, dist_high_space, angle)

            self.embedding_ = np.vstack((self.embedding_, np.array([x, y])))

            test1 = np.array(euclidean_distances(self.embedding_)[-1])
            test2 = np.array(euclidean_distances(random_config)[-1])
            stress = ((test1 - test2) ** 2).sum() / 2

            print("Funktion fertig")

        assert(random_config.shape == confi_matrix.shape)
        assert(random_config.shape[0] == self.embedding_.shape[0])
        return self.embedding_

    def fit_transform(self, X, y=None, init=None):
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
