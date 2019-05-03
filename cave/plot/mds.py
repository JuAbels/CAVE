"""
Multi-dimensional Scaling (MDS) with SMACOF Algorithm and Classical method with extendig
mds to new points.

Done as Undergraduate Thesis. Adopted by scikit-learn.
<https://github.com/scikit-learn/scikit-learn/pull/9834/files>
"""

import numpy as np
from numpy import linalg as la
import warnings
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.manifold.mds import smacof


class MDS_BA(BaseEstimator):
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

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None, logger=None):
        """
        Computes eigenvalues and eigenvectors of the normalized distance matrix.
        Calculate the normalized distance matrix with double centering formula.

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
            n = len(D)                               # dimension of distance matrix
            D_sq = np.square(D)                      # squared of distance matrix
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

    def fit_transform(self, X_train, X_New=None, init=None, logger=None):
        """
        Fit the data from X, and returns the embedded coordinates. Depending
        on self.method, the computations with the MDS are with SMACOF algorithm
        or with classical method.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                if dissimilarity='precomputed'

        init : ndarray, shape (n_samples,), optional, default: None
            Should only be used with the non-inductive MDS (SMACOF).
            If None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array.

        Returns
        -------
        ret: np.array, shape (n_samples + m_samples, 2)
            the 2-dimensional coordinates of all embedding configurations.
        """
        if self.method == 'inductive':
            if init is not None:
                raise ValueError("Init is only for the non-inductive MDS.")
            ret = self._fit_transform_ext(X_train, X_New, logger=logger)
        else:
            ret = self._fit_transform(X_train, init, logger=logger)
        return ret

    def transform(self, X, logger=None):
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
            New coordinates of the extending points.

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

    def embedding(self, k):
        """
        Computes the position of the points in the embedding space.

        embedding = matrix_eigenvectors * diagonal_matrix_eigenvalues^(1/2)

        Parameters:
        ----------
        k: int
            Size of new dimension.

        Returns:
        -------
        embedding: np.array, shape (n_samples, 2)
        """
        embedding = np.dot(self.e_vecs_[:, :k], np.diag(np.sqrt(self.e_vals_[:k])))
        return embedding

    def _fit_transform(self, X, y=None, init=None, logger=None):
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

        Returns:
        -------
        embedding_: np.array, shape(n_samples + m_samples, 2)
            New coordinates of all configurations. Results of non-metric method.
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

        if logger:
            logger.info("Multidimensional scaling with SMACOF algorithm")
            logger.info("New koordinates: %s" % (self.embedding_))
            logger.info("Stress Value: %s" % (self.stress_))

        return self.embedding_

    def _fit_transform_ext(self, X_train, X_new=None, logger=None):
        """
        Calculate coordinates of embedding space with classical method. If, in addition to
        the random configurations, there are local configurations, extend the embedding space
        with an expansion of MDS.

        Parameters:
        ----------
        X_train: np.array, shape (n_samples, n_samples)
            Distance matrix of all random configurations.

        X_new: np.array, shape (m_samples, n_samples)
            Distance matrix where each row corresponds to a new point and each column to
            the corresponding distance to a random point.

        Returns:
        -------
        dists: np.arrray, shape (n_samples + m_samples, 2)
            Coordinates of embedding of random and local configurations.
        """
        # Make fitting with distances of random configurations
        self.fit(X_train)

        # Transform the random configurations to two dimensions with eigenvectors and eigenvalues
        training_points = self.embedding(2)

        if X_new is None:
            if logger:
                logger.info("Only calculated embedding of random configurations, without extending points")
                logger.info("New coordinates: %s" % (training_points))
                logger.info("Finished with computation of new coordinates")
            return training_points

        # Calculate center similarities of local and random distance matrix
        new_traing_dists = self.center_similarities(X_new, X_train, logger=logger)

        # Transform the new points down in embedding space
        new_points = self.transform(new_traing_dists)

        dists = np.vstack((training_points, new_points))

        if logger:
            logger.info("New coordinates: %s" % (dists))
            logger.info("Finished with computation of new coordinates")

        return dists

    def center_similarities(self, D_aX, D_XX, logger=None):
        """
        Centers similarities D_aX around D_XX.

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

        if logger:
            logger.info("Matrix of center similarities of training and new points: %s" % (new_similarities))

        return new_similarities

    def _mds_project(self, new_similarities, k, logger=None):
        """
        Make the projection of the new points down in the embedding space.

        Parameters
        ----------
        new_similarities: np.array, shape (m_samples, n_samples)
            Normalized Matrix of local and random distance matrix.

        k: int
            Number of dimensions of the embedding space.

        Returns
        -------
        e_projections: np.array, shape (m_samples, 2)
            Coordinates of local configurations in embedding space.
        """

        '''
        e_projections = np.zeros((len(new_similarities), k))
        
        for i in range(len(new_similarities)):
            for j in range(k):
                e_projections[i, j] = ((np.dot(self.e_vecs_[:, j],
                                               new_similarities[i]) /
                                        np.sqrt(self.e_vals_[j])))
        '''

        e_projections = np.dot(new_similarities, np.dot(self.e_vecs_[:, :k],
                                        np.diag(1/np.sqrt(self.e_vals_[:k]))))

        return e_projections
