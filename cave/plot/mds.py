"""
Multi-dimensional Scaling (MDS) with SMACOF Algorithm.

Done as Undergraduate Thesis. Adopted by scikit-learn.
"""

import numpy as np
from inspect import signature
from collections import defaultdict
import warnings
import numbers

from joblib import Parallel, delayed
import scipy.sparse as sp


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class BaseEstimator:
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class IsotonicRegression(BaseEstimator, TransformerMixin):
    """Isotonic regression model.

    The isotonic regression optimization problem is defined by::

        min sum w_i (y[i] - y_[i]) ** 2

        subject to y_[i] <= y_[j] whenever X[i] <= X[j]
        and min(y_) = y_min, max(y_) = y_max

    where:
        - ``y[i]`` are inputs (real numbers)
        - ``y_[i]`` are fitted
        - ``X`` specifies the order.
          If ``X`` is non-decreasing then ``y_`` is non-decreasing.
        - ``w[i]`` are optional strictly positive weights (default to 1.0)

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y_min : optional, default: None
        If not None, set the lowest value of the fit to y_min.

    y_max : optional, default: None
        If not None, set the highest value of the fit to y_max.

    increasing : boolean or string, optional, default: True
        If boolean, whether or not to fit the isotonic regression with y
        increasing or decreasing.

        The string value "auto" determines whether y should
        increase or decrease based on the Spearman correlation estimate's
        sign.

    out_of_bounds : string, optional, default: "nan"
        The ``out_of_bounds`` parameter handles how x-values outside of the
        training domain are handled.  When set to "nan", predicted y-values
        will be NaN.  When set to "clip", predicted y-values will be
        set to the value corresponding to the nearest train interval endpoint.
        When set to "raise", allow ``interp1d`` to throw ValueError.


    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    Notes
    -----
    Ties are broken using the secondary method from Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    Leeuw, Psychometrica, 1977
    """
    def __init__(self, y_min=None, y_max=None, increasing=True,
                 out_of_bounds='nan'):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.")
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array


def effective_n_jobs(n_jobs=-1):
    """Determine the number of jobs that can actually run in parallel

    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1
    means requesting all available workers for instance matching the number of
    CPU cores on the worker host(s).

    This method should return a guesstimate of the number of workers that can
    actually perform work concurrently with the currently enabled default
    backend. The primary use case is to make it possible for the caller to know
    in how many chunks to slice the work.

    In general working on larger data chunks is more efficient (less scheduling
    overhead and better use of CPU cache prefetching heuristics) as long as all
    the workers have enough work to do.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10

    """
    # backend, _ = get_active_backend()
    # return backend.effective_n_jobs(n_jobs=n_jobs)
    pass


def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None):
    """Computes multidimensional scaling using SMACOF algorithm

    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.

    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress.
    """
    # TODO check symmetric
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)  # checken der symmetrie, TODO

    n_samples = dissimilarities.shape[0]  # Anzahl der Configurations
    random_state = check_random_state(random_state)  # erstelle aus ein RandomState Object aus Int random state, TODO

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()  # Multipliziere diss. mit Matrix die groß ist und
                                                                    # ab diagonale nach unten nur 0en & darüber 1en
                                                                    # Aus Ergenis wird ein Vector
    sim_flat_w = sim_flat[sim_flat != 0]  # entferne aus der Vector alle 0en
    if init is None:  # Erstellen von einer Random Configuration
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)  # Erstelle aus RandomState ein Vector
        X = X.reshape((n_samples, n_components))  # Erstelle daraus eine Matrix
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()  # fit line in data, die nicht linear sein muss -> TODO Why?
    for it in range(max_iter):
        # Compute distance and monotonic regression
        # dis = euclidean_distances(X)
        dis = dissimilarities  # das ist ja eigentlich schon implementiert mit get_distance?

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)  # TODO WAS wird hier geamcht?
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1


def smacof(dissimilarities, metric=True, n_components=2, init=None, n_init=8,
           n_jobs=None, max_iter=300, verbose=0, eps=1e-3, random_state=None,
           return_n_iter=False):
    """Computes multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can summarized by the following steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress. -> TODO: beachten also of nonmetric algorithm? Why

    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : boolean, optional, default: True
        Compute metric or nonmetric SMACOF algorithm.

    n_components : int, optional, default: 2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray, shape (n_samples, n_components), optional, default: None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, optional, default: 8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, optional, default: 0
        Level of verbosity.

    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        The generator used to initialize the centers.  If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    return_n_iter : bool, optional, default: False
        Whether or not to return the number of iterations.

    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    Notes
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    # dissimilarities = check_array(dissimilarities)   # By default, the input is checked to be a non-empty 2D array
                                                       # containing only finite values
    random_state = check_random_state(random_state)  # Create from int a Object of RadomState

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None

    # if effective_n_jobs(n_jobs) == 1:
    if n_jobs is None:
        for it in range(n_init):
            pos, stress, n_iter_ = _smacof_single(
                dissimilarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_smacof_single)(
                dissimilarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed)
            for seed in seeds)
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class MDS_New(BaseEstimator):
    """Multidimensional scaling

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
                ``fit_transform

        Attributes
        ----------
        embedding_ : array-like, shape (n_samples, n_components)
            Stores the position of the dataset in the embedding space.

        stress_ : float
            The final value of the stress (sum of squared distance of the
            disparities and the distances for all constrained points).
    """
    def __init__(self, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=None,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components      # call with 2
        self.dissimilarity = dissimilarity    # "precomputed"
        self.metric = metric  # True -> Metric, False -> Nonmetric
        self.n_init = n_init  # how often SMADOF with diff. init. End: take run with smallest stress
        self.max_iter = max_iter  # max number of iteration
        self.eps = eps  # tolerence with respect to stress -> to declare convergence
        self.verbose = verbose  # Grad der Ausführlichkeit
        self.n_jobs = n_jobs  # number of jobs, if n_init > 1 -> parallel jobs
        self.random_state = random_state      # random_state=12345

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
        # X = check_array(X)

        # Lieber hier ein Error?
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity != "precomputed" and self.dissimilarity != "euclidean":
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_ = smacof(X, metric=self.metric,
            n_components=self.n_components, init=init, n_init=self.n_init,
            n_jobs=self.n_jobs, max_iter=self.max_iter, verbose=self.verbose,
            eps=self.eps, random_state=self.random_state,
            return_n_iter=True)

        return self.embedding_
