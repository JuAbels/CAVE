#!/bin/python

__author__ = "Marius Lindauer & Joshua Marben"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Joshua Marben"
__email__ = "marbenj@cs.uni-freiburg.de"

import os
import sys
import inspect
import logging
import copy
import time
from math import sqrt

import numpy as np
# from sklearn.manifold.mds import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, BasicTicker, CustomJS, Slider
from bokeh.models.sources import CDSView
from bokeh.models.filters import GroupFilter, BooleanFilter
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import CheckboxButtonGroup, CheckboxGroup, RadioButtonGroup, Button, Select, Div

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))  # noqa
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))  # noqa
if cmd_folder not in sys.path:  # noqa
    sys.path.append(cmd_folder)  # noqa

from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.optimizer.objective import average_cost
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.configspace import ConfigurationSpace
from ConfigSpace.util import impute_inactive_values
from ConfigSpace import CategoricalHyperparameter

from cave.utils.convert_for_epm import convert_data_for_epm
from cave.utils.helpers import escape_parameter_name, get_config_origin, combine_runhistories
from cave.utils.helpers import create_random_runhistories, combine_random_local, create_new_rhs  # Julia BA
from cave.utils.timing import timing
from cave.utils.io import export_bokeh
from cave.plot.mds import MDS_BA
from sklearn.metrics import euclidean_distances


from tempfile import TemporaryFile

from cave.utils.bokeh_routines import get_checkbox, get_radiobuttongroup


class ConfiguratorFootprintPlotter(object):

    def __init__(self,
                 scenario: Scenario,
                 rhs: RunHistory,
                 incs: list=None,
                 final_incumbent=None,
                 rh_labels=None,
                 max_plot: int=-1,
                 contour_step_size=0.2,
                 use_timeslider: bool=False,
                 num_quantiles: int=10,
                 timeslider_log: bool=True,
                 output_dir: str=None,
                 reduction_method: str="classic"
                 ):
        '''
        Creating an interactive plot, visualizing the configuration search space.
        The runhistories are correlated to the individual runs.
        Each run consists of a runhistory (in the smac-format), a list of incumbents
        If the dict "additional_info" in the RunValues of the runhistory contains a nested dict with
        additional_info["timestamps"]["finished"], using those timestamps to sort data

        Parameters
        ----------
        scenario: Scenario
            scenario
        rhs: List[RunHistory]
            runhistories from configurator runs, only data collected during optimization (no validation!)
        incs: List[List[Configuration]]
            incumbents per run, last entry is final incumbent
        final_incumbent: Configuration
            final configuration (best of all runs)
        max_plot: int
            maximum number of configs to plot, if -1 plot all
        contour_step_size: float
            step size of meshgrid to compute contour of fitness landscape
        use_timeslider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        num_quantiles: int
            number of quantiles for the slider/ number of static pictures
        timeslider_log: bool
            whether to use a logarithmic scale for the timeslider/quantiles
        output_dir: str
            output directory
        reduction_method: str
            selection of which method is used for dimensionreduction
        '''
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        self.scenario = scenario
        self.rhs_ = rhs

        self.reduction_method = reduction_method
        self.logger.info("Dimension reduction is carried out with %s method." % self.reduction_method)

        self.combined_rh_ = combine_runhistories(self.rhs_)  # TODO: BA Julia
        self.random_rh, self.local_rh = create_random_runhistories(self.combined_rh_)  # Julia BA -> need for get_mds

        # create correct combined rh with firstly the random configurations and then residual confis
        self.combined_rh = combine_random_local(self.random_rh, self.local_rh, self.logger)
        self.rhs = create_new_rhs(self.rhs_)  # for creating new rhs with correct combination

        # Case for only random Configurations
        # self.rhs = []
        # for i in self.rhs_:
        #    random, local = create_random_runhistories(i)
        #    self.rhs.append(random)
        # self.combined_rh = self.random_rh
        # self.local_rh = self.random_rh

        for ele in self.random_rh.data:
            assert(self.combined_rh.ids_config[ele[0]] == self.random_rh.ids_config[ele[0]])

        self.incs = incs
        self.rh_labels = rh_labels if rh_labels else [str(idx) for idx in range(len(self.rhs))]
        self.max_plot = max_plot
        self.use_timeslider = use_timeslider
        self.num_quantiles = num_quantiles
        self.contour_step_size = contour_step_size
        self.output_dir = output_dir
        self.timeslider_log = timeslider_log

        # Preprocess input
        self.default = scenario.cs.get_default_configuration()
        self.final_incumbent = final_incumbent

        # Julia BA
        self.configs_in_run = {label: rh.get_all_configs_combined() for label, rh in zip(self.rh_labels, self.rhs)}

    def run(self):
        """
        Uses available Configurator-data to perform a MDS, estimate performance
        data and plot the configurator footprint.
        """
        default = self.scenario.cs.get_default_configuration()

        self.logger.info("Number of configurations: %s" % len(self.combined_rh.config_ids))
        # self.combined_rh = self.reduce_runhistory(self.combined_rh, self.max_plot, keep=[a for b in self.incs for a in b]+[default])
        self.combined_rh = self.reduce_runhistory(self.combined_rh, 1000**2,
                                                  keep=[a for b in self.incs for a in b] + [default])
        self.logger.info("Number of configurations: %s" % len(self.combined_rh.config_ids))

        if len(self.combined_rh.data) < len(self.combined_rh_.data):
            self.random_rh, self.local_rh = create_random_runhistories(self.combined_rh)
            number = len(self.random_rh.get_all_configs_combined())
            # Test if local and random rh have same combination as combined
            for ele in range(number):
                assert(self.random_rh.get_all_configs_combined()[ele] == self.combined_rh.get_all_configs_combined()[ele])
            for ele in range(len(self.local_rh.get_all_configs_combined())):
                assert (self.local_rh.get_all_configs_combined()[ele] == self.combined_rh.get_all_configs_combined()[ele+number])

        # Julia BA
        conf_matrix, conf_list, runs_per_quantile, timeslider_labels, rand_index = self.get_conf_matrix(self.combined_rh, self.incs)
        # np.savetxt("matrix_cplex.txt", conf_matrix)

        dists = self.get_distance(conf_matrix, self.scenario.cs)
        red_dists = self.call_method(method=self.reduction_method,
                                     combined=dists,
                                     rand_confis=rand_index)
                                     # rand_confis=len(conf_matrix))

        assert(conf_matrix.shape[0] == red_dists.shape[0])
        cost_value = self.calculate_costvalue(dists, red_dists)
        self.logger.info("Calculate costvalue of distances in high and low dimensional: %f" % cost_value)

        contour_data = {}

        self.logger.info("Call of get_pred_surface combined")
        # contour_data['combined'] = self.get_pred_surface(self.combined_rh, X_scaled=red_dists,  # Here call with X, y
        #                                                  conf_list=copy.deepcopy(conf_list),
        #                                                  contour_step_size=self.contour_step_size)
        if not any([label.startswith('budget') for label in self.rh_labels]):
            contour_data['combined'] = self.get_pred_surface(self.combined_rh, X_scaled=red_dists,
                                                             conf_list=copy.deepcopy(conf_list),
                                                             contour_step_size=self.contour_step_size)
        for label, rh in zip(self.rh_labels, self.rhs):
            self.logger.info("Call of get_pred_surface")
            contour_data[label] = self.get_pred_surface(rh, X_scaled=red_dists,
                                                        conf_list=copy.deepcopy(conf_list),
                                                        contour_step_size=self.contour_step_size)

        print("Finished label rh")

        return self.plot(red_dists,
                         conf_list,
                         runs_per_quantile,
                         inc_list=self.incs,
                         contour_data=contour_data,
                         use_timeslider=self.use_timeslider,
                         timeslider_labels=timeslider_labels)

    def call_method(self,
                    method,
                    rand_confis,
                    combined=None):
                    #conf_matrix_random=None,
                    #conf_matrix_local=None):
        """
        Helpfunction to call mds with the selected method.

        Parameters
        ----------
        method: str
            depending on which method, the distance is determined
        combined: np.array, shape (n_samples + m_samples, n_samples + m_samples)
            full matrix of distances between all configurations, required for SMACOF-Algorithm
        conf_matrix_random: np.array, shape (n_samples, k_features)
            matrix of random configurations
        conf_matrix_random: np.array, shape (m_samples, k_features)
            matrix of local configurations

        Returns
        -------
        dists: np.array(n_samples, 2)
            coordinates of all configurations in embedding space.
        """
        if method == "smacof":
            return self.get_mds(combined=combined, classical_method=False, logger=self.logger)
        # TODO elif method == "autoencoder"

        random_dists = combined[:rand_confis, :rand_confis]
        train_new_dists = combined[rand_confis:, :rand_confis]
        return self.get_mds(combined=combined, random_dists=random_dists,
                            train_and_new=train_new_dists, classical_method=True, logger=self.logger)

    @timing
    def get_pred_surface(self, rh, X_scaled, conf_list: list, contour_step_size):
        """fit epm on the scaled input dimension and
        return data to plot a contour plot of the empirical performance

        Parameters
        ----------
        rh: RunHistory
            runhistory
        X_scaled: np.array
            configurations in scaled 2dim
        conf_list: list
            list of Configuration objects
        contour_step_size: float
            step-size for contour

        Returns
        -------
        contour_data: (np.array, np.array, np.array)
            x, y, Z for contour plots
        """
        # use PCA to reduce features to also at most 2 dims
        scen = copy.deepcopy(self.scenario)  # pca changes feats
        if scen.feature_array.shape[1] > 2:
            self.logger.debug("Use PCA to reduce features to from %d dim to 2 dim", scen.feature_array.shape[1])
            # perform PCA
            insts = scen.feature_dict.keys()
            feature_array = np.array([scen.feature_dict[i] for i in insts])
            feature_array = StandardScaler().fit_transform(feature_array)
            feature_array = PCA(n_components=2).fit_transform(feature_array)
            # inject in scenario-object
            scen.feature_array = feature_array
            scen.feature_dict = dict([(inst, feature_array[idx, :]) for idx, inst in enumerate(insts)])
            scen.n_features = 2

        # convert the data to train EPM on 2-dim featurespace (for contour-data)
        self.logger.debug("Convert data for epm.")

        # X: matrix, input dimension -> so dim n, X matrix with configuartion x features for all observed samples
        # y: vector, y matrix with all observations
        X, y, types = convert_data_for_epm(scenario=scen, runhistory=rh, logger=self.logger)

        types = np.array(np.zeros((2 + scen.feature_array.shape[1])), dtype=np.uint)
        num_params = len(scen.cs.get_hyperparameters())

        # impute missing values in configs and insert MDS'ed (2dim) configs to the right positions
        conf_dict = {}
        # Remove forbidden clauses (this is necessary to enable the impute_inactive_values-method, see #226)
        cs_no_forbidden = copy.deepcopy(conf_list[0].configuration_space)
        cs_no_forbidden.forbidden_clauses = []
        for idx, c in enumerate(conf_list):
            c.configuration_space = cs_no_forbidden
            conf_list[idx] = impute_inactive_values(c)
            conf_dict[str(conf_list[idx].get_array())] = X_scaled[idx, :]

        print("Conf_dict created, now try to create X_trans")

        X_trans = []
        idx = 0
        for x in X:
            x_scaled_conf = conf_dict[str(x[:num_params])]
            # append scaled config + pca'ed features (total of 4 values) per config/feature-sample
            X_trans.append(np.concatenate((x_scaled_conf, x[num_params:]), axis=0))
            idx += 1
        X_trans = np.array(X_trans)

        self.logger.debug("Train random forest for contour-plot.")
        bounds = np.array([(0, np.nan), (0, np.nan)], dtype=object)
        model = RandomForestWithInstances(types=types, bounds=bounds,
                                          instance_features=np.array(scen.feature_array),
                                          ratio_features=1.0)

        start = time.time()
        model.train(X_trans, y)
        self.logger.debug("Fitting random forest took %f time", time.time() - start)

        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, contour_step_size),
                             np.arange(y_min, y_max, contour_step_size))

        self.logger.debug("x_min: %f, x_max: %f, y_min: %f, y_max: %f", x_min, x_max, y_min, y_max)
        self.logger.debug("Predict on %d samples in grid to get surface (step-size: %f)",
                          np.c_[xx.ravel(), yy.ravel()].shape[0], contour_step_size)

        start = time.time()
        Z, _ = model.predict_marginalized_over_instances(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        self.logger.debug("Predicting random forest took %f time", time.time() - start)

        return xx, yy, Z

    @timing
    def get_distance(self, conf_matrix, cs: ConfigurationSpace):
        """
        Computes the distance between all pairs of configurations.

        Parameters
        ----------
        conf_matrix: np.array
            numpy array with cols as parameter values
        cs: ConfigurationSpace
            ConfigurationSpace to get conditionalities

        Returns
        -------
        dists: np.array
            np.array with distances between configurations i,j in dists[i,j] or dists[j,i]
        """
        self.logger.debug("Calculate distance between configurations.")
        n_confs = conf_matrix.shape[0]
        dists = np.zeros((n_confs, n_confs))

        is_cat = []
        depth = []
        for _, param in cs._hyperparameters.items():
            if type(param) == CategoricalHyperparameter:
                is_cat.append(True)
            else:
                is_cat.append(False)
            depth.append(self.get_depth(cs, param))
        is_cat = np.array(is_cat)
        depth = np.array(depth)

        # TODO tqdm
        start = time.time()
        for i in range(n_confs):
            for j in range(i + 1, n_confs):
                dist = np.abs(conf_matrix[i, :] - conf_matrix[j, :])  # get distance between two

                # iterate over nan positions and depending on nan and not nan insert 1 or 0 as distance
                # index_nan = [nan for nan in range(dist.shape[0]) if np.isnan(dist[nan])]
                for index in [nan for nan in range(dist.shape[0]) if np.isnan(dist[nan])]:
                    if np.isnan(conf_matrix[i, index]) and np.isnan(conf_matrix[j, index]):
                        dist[index] = 0
                        continue
                    dist[index] = 1

                dist[np.logical_and(is_cat, dist != 0)] = 1

                dist = np.square(dist/depth)              # TODO: Get_distance methodic
                dist = sqrt(np.sum(dist))

                print(dist)
                # dist = np.sum(dist / depth)
                dists[i, j] = dist
                dists[j, i] = dist
            if 5 < n_confs and i % (n_confs // 5) == 0:
                self.logger.debug("%.2f%% of all distances calculated in %.2f seconds...", 100 * i / n_confs,
                                  time.time() - start)

        return dists

    @timing
    def get_distance_random_new(self, train_matrix, new_matrix, cs: ConfigurationSpace):
        """
        Computes the distance between random and new configurations in original space.

        Parameters
        ----------
        train_matrix: np.array, shape (n_samples, n_features)
            numpy array of the random configurations.
        new_matrix: np.array, shape (m_samples, m_features)
             numpy array of the residual configurations of combined_rh.
        cs: ConfigurationSpace
            ConfigurationSpace to get conditionalities

        Returns
        -------
        dists: np.array, shape (m_samples, n_samples)
            np.array with distances between each new point and all random configurations.
        """
        self.logger.debug("Calculate distance between random and local configurations.")
        local_confs = new_matrix.shape[0]
        random_confs = train_matrix.shape[0]
        dists = np.zeros((local_confs, random_confs))

        is_cat = []
        depth = []
        for _, param in cs._hyperparameters.items():
            if type(param) == CategoricalHyperparameter:
                is_cat.append(True)
            else:
                is_cat.append(False)
            depth.append(self.get_depth(cs, param))
        is_cat = np.array(is_cat)
        depth = np.array(depth)

        # TODO tqdm
        start = time.time()
        for i in range(local_confs):
            for j in range(random_confs):
                dist = np.abs(train_matrix[j, :] - new_matrix[i, :])
                # iterate over nan positions and depending on nan and not nan insert 1 or 0 as distance
                for index in [nan for nan in range(dist.shape[0]) if np.isnan(dist[nan])]:
                    if np.isnan(train_matrix[j, index]) and np.isnan(new_matrix[i, index]):
                        dist[index] = 0
                        continue
                    dist[index] = 1
                dist[np.logical_and(is_cat, dist != 0)] = 1
                dist /= depth
                dists[i, j] = np.sum(dist)
            if 5 < local_confs and i % (local_confs // 5) == 0:
                self.logger.debug("%.2f%% of all distances calculated in %.2f seconds...", 100 * i / local_confs,
                                  time.time() - start)

        return dists

    def calculate_costvalue(self, dists, red_dists):
        """
        Helpfunction to calculate the costvalue to test how big the difference of distance is in the embedding
        and original space.

        Parameters
        ----------
        dists: np.array, shape(n_samples, n_samples)
            Matrix of the distances in the original space.
        red_dists: np.array, shape(n_samples, k_dimensions)
            Koordinates o

        Rreturns
        --------
        costvalue: float
            Costvalue of the distances of the two spaces.

            costvalues = sum_i sum_j=i+1 (distance_low_space_ij - distance_high_space_ij)
        """
        n_conf = dists.shape[0]
        low_dists = euclidean_distances(red_dists)
        costvalue = []
        mean_actual = []

        self.logger.info("Low_distances: \n %s" % low_dists)

        for i in range(n_conf - 1):
            for j in range(i+1, n_conf):
                costvalue.append((dists[i][j] - low_dists[i][j])**2)
                mean_actual.append(low_dists[i][j])

        mean_actual_value = sum(mean_actual) / len(mean_actual)
        actual = [(mean_actual_value - dif)**2 for dif in mean_actual]
        pred_actual = sum(costvalue)
        rse = pred_actual / sum(actual)

        self.logger.info("Costvalue calculated with RSE: %s" % rse)

        costvalue = sum(costvalue) / len(costvalue)

        return costvalue

    def get_depth(self, cs: ConfigurationSpace, param: str):
        """
        Get depth in configuration space of a given parameter name
        breadth search until reaching a leaf for the first time

        Parameters
        ----------
        cs: ConfigurationSpace
            ConfigurationSpace to get parents of a parameter
        param: str
            name of parameter to inspect
        """
        parents = cs.get_parents_of(param)
        if not parents:
            return 1
        new_parents = parents
        d = 1
        while new_parents:
            d += 1
            old_parents = new_parents
            new_parents = []
            for p in old_parents:
                pp = cs.get_parents_of(p)
                if pp:
                    new_parents.extend(pp)
                else:
                    return d

    @timing
    def get_mds(self, combined=None, random_dists=None, train_and_new=None, classical_method=True, logger=None):
        """
        Compute multi-dimensional scaling -- using classical method with extending new points or nonmetrical method.

        Parameters
        ----------
        combined: np.array, shape (n_samples + m_samples, n_samples + m_samples)
            full matrix of distances between all configurations, required for SMACOF-Algorithm

        random_dists: np.array, shape (n_samples, n_samples)
            matrix of distances between all random configurations

        train_and_new: np.array, shape (m_samples, n_samples)
            distance matrix of each new point and all random configurations

        classic_method: bool
            If True perfom MDS with classical method, otherwise with SMACOF-Algorithm.

        Returns
        -------
        dists: np.array, shape (n_samples + m_samples. 2)
            scaled coordinates in 2-dim room

        Examples
        --------
        >>> from cave.plot.mds import MDS_BA
        >>> import numpy as np

        >>> original_space = np.array([[0, 2, 4, 3], [2, 0, 7, 9], [4, 7, 0, 4], [3, 9, 4, 0]])
        >>> mds = MDS_BA(n_components=2, dissimilarity="precomputed", random_state=12345, method='inductive')
        >>> new_coordinates = mds.fit_transform(original_space)
        >>> new_coordinates
        array([[ 0.8497362 ,  1.1287592 ],
               [ 4.94371831, -0.02744067],
               [-1.84412041, -2.45721537],
               [-3.9493341 ,  1.35589684]])

        >>> extending_points =  np.array([[4, 5, 2, 1], [3, 8, 3, 6]])
        >>> new_coordinates = mds.fit_transform(original_space, extending_points)
        >>> new_coordinates
        array([[ 0.8497362 ,  1.1287592 ],
               [ 4.94371831, -0.02744067],
               [-1.84412041, -2.45721537],
               [-3.9493341 ,  1.35589684],
               [-1.08794886, -0.84492993],
               [-1.53415797, -2.27636909]])
        """
        # TODO: Here for BA Julia
        # TODO there are ways to extend MDS to provide a transform-method. if
        #   available, train on randomly sampled configs and plot all
        # TODO MDS provides 'n_jobs'-argument for parallel computing...

        self.logger.info("Start Multidimensional Scaling.")

        if classical_method is False:

            self.logger.info("Perform Multidimesnional Scaling with SMACOF-Algorithm")
            smacof = MDS_BA(n_components=2, dissimilarity="precomputed", random_state=12345)
            dists = smacof.fit_transform(combined, logger=logger)
            self.logger.warning(dists)
            self.logger.info("Dimension of embedding space: %s" % (dists.shape[0]))

            return dists

        self.logger.info("Perform Multidimesnional Scaling with classical method")
        mds = MDS_BA(n_components=2, dissimilarity="precomputed", random_state=12345, method='inductive')

        dists = mds.fit_transform(random_dists, train_and_new, logger=logger)
        self.logger.info("Dimension of embedding space: %s" % (dists.shape[0]))

        return dists

    def reduce_runhistory(self,
                          rh: RunHistory,
                          max_configs: int,
                          keep=None):
        """
        Reduce configs to desired number, by default just drop the configs with the fewest runs.

        Parameters
        ----------
        rh: RunHistory
            runhistory that is to be reduced
        max_configs: int
            if > -1 reduce runhistory to at most max_configs
        keep: List[Configuration]
            list of configs that should be kept for sure (e.g. default, incumbents)

        Returns
        -------
        rh: RunHistory
            reduced runhistory
        """
        configs = rh.get_all_configs_combined()  # Julia BA
        self.logger.info("MAX_configs: %s" % max_configs)
        self.logger.info("len configs: %s" % len(configs))
        if max_configs <= 0 or max_configs > len(configs):  # keep all
            self.logger.info("Not calling function reduce runhistory")
            return rh

        self.logger.info("calling function reduce runhistory")
        runs = [(c, len(rh.get_runs_for_config(c))) for c in configs]
        if not keep:
            keep = []
        runs = sorted(runs, key=lambda x: x[1])[-self.max_plot:]
        keep = [r[0] for r in runs] + keep
        self.logger.info("Reducing number of configs from %d to %d, dropping from the fewest evaluations",
                         len(configs), len(keep))

        new_rh = RunHistory(average_cost)
        for k, v in list(rh.data.items()):
            c = rh.ids_config[k.config_id]
            if c in keep:
                new_rh.add(config=rh.ids_config[k.config_id],
                           cost=v.cost, time=v.time, status=v.status,
                           instance_id=k.instance_id, seed=k.seed,
                           additional_info=v.additional_info)  # added from Julia
        return new_rh

    @timing
    def get_conf_matrix(self, rh, incs):
        """
        Iterates through runhistory to get a matrix of configurations (in
        vector representation), a list of configurations and the number of
        runs per configuration in a quantiled manner.

        Parameters
        ----------
        rh: RunHistory
            smac.runhistory
        incs: List[List[Configuration]]
            incumbents of configurator runs, last entry is final incumbent

        Returns
        -------
        conf_matrix: np.array
            matrix of configurations in vector representation
        conf_list: np.array
            list of all Configuration objects that appeared in runhistory
            the order of this list is used to determine all kinds of properties
            in the plotting (but is arbitrarily determined)
        runs_per_quantile: np.array
            numpy array of runs per configuration per quantile
        labels: List[str]
            labels for timeslider (i.e. wallclock-times)
        """
        conf_list = []
        conf_matrix = []
        # Get all configurations. Index of c in conf_list serves as identifier

        for c in rh.get_all_configs_combined():  # Julia BA
            if c not in conf_list:
                conf_matrix.append(c.get_array())
                conf_list.append(c)

        # for inc in [a for b in incs for a in b]:
            # if inc not in conf_list:
                # conf_matrix.append(inc.get_array())
                # conf_list.append(inc)

        # Julia BA
        index = len(self.random_rh.get_all_configs_combined())
        for inc in [a for b in incs for a in b]:
            if inc not in conf_list:
                origin = get_config_origin(inc)
                if origin == "Random":
                    conf_matrix.insert(index, inc.get_array())
                    index += 1
                else:
                    conf_matrix.append(inc.get_array())
                conf_list.append(inc)

        # Sanity check, number quantiles must be smaller than the number of configs
        if self.num_quantiles >= len(conf_list):
            self.logger.info("Number of quantiles %d bigger than number of configs %d, reducing to %d quantiles",
                              self.num_quantiles, len(conf_list), len(conf_list) - 1)
            self.num_quantiles = len(conf_list) - 1

        # We want to visualize the development over time, so we take
        # screenshots of the number of runs per config at different points
        # in (i.e. different quantiles of) the runhistory, LAST quantile
        # is full history!!
        labels, runs_per_quantile = self._get_runs_per_config_quantiled(rh, conf_list, quantiles=self.num_quantiles)
        assert(len(runs_per_quantile) == self.num_quantiles)

        # Get minimum and maximum for sizes of dots
        self.min_runs_per_conf = min([i for i in runs_per_quantile[-1] if i > 0])
        self.max_runs_per_conf = max(runs_per_quantile[-1])
        self.logger.debug("Min runs per conf: %d, Max runs per conf: %d", self.min_runs_per_conf, self.max_runs_per_conf)
        self.logger.debug("Gathered %d configurations from 1 runhistories." % len(conf_list))

        runs_per_quantile = np.array([np.array(run) for run in runs_per_quantile])
        return np.array(conf_matrix), np.array(conf_list), runs_per_quantile, labels, index

    @timing
    def get_conf_matrix_random_local(self, rh_radom, rh_local, incs):
        """Returns the matrix of random_rh and local_rh."""
        conf_list = []
        conf_matrix_random = []
        conf_matrix_local = []

        for c in rh_radom.get_all_configs_combined():
            if c not in conf_list:
                conf_matrix_random.append(c.get_array())
                conf_list.append(c)

        for c in rh_local.get_all_configs_combined():
            if c not in conf_list:
                conf_matrix_local.append(c.get_array())
                conf_list.append(c)

        for inc in [a for b in incs for a in b]:
            if inc not in conf_list:
                if get_config_origin(inc) == 'Random':
                    conf_matrix_random.append(inc.get_array())
                elif get_config_origin(inc) == 'Acquisition Function' or get_config_origin(inc) == 'Unknown':
                    conf_matrix_local.append(inc.get_array())
                conf_list.append(inc)

        assert(len(conf_matrix_local) + len(conf_matrix_random) == len(conf_list))
        return np.array(conf_matrix_random), np.array(conf_matrix_local)

    @timing
    def _get_runs_per_config_quantiled(self, rh, conf_list, quantiles):
        """Returns a list of lists, each sublist representing the current state
        at that timestep (quantile). The current state means a list of times
        each config was evaluated at that timestep.

        Parameters
        ----------
        rh: RunHistory
            rh to be split up
        conf_list: list
            list of all Configuration objects that appear in runhistory
        quantiles: int
            number of fractions to split rh into

        Returns:
        --------
        labels: List[str]
            labels for timeslider (i.e. wallclock-times)
        runs_per_quantile: np.array
            numpy array of runs per configuration per quantile
        """
        runs_total = len(rh.data)
        # Iterate over the runhistory's entries in ranges and creating each
        # sublist from a "snapshot"-runhistory
        labels, last_time_seen = [], -1  # label, means wallclocktime at splitting points
        r_p_q_p_c = []  # runs per quantile per config
        as_list = list(rh.data.items())
        scale = np.geomspace if self.timeslider_log else np.linspace

        # Trying to work with timestamps if they are available
        timestamps = None
        try:
            as_list = sorted(as_list, key=lambda x: x[1].additional_info['timestamps']['finished'])
            timestamps = [x[1].additional_info['timestamps']['finished'] for x in as_list]
            time_ranges = scale(timestamps[0], timestamps[-1], num=quantiles+1, endpoint=True)
            ranges = []
            idx = 0
            for time_idx, time in enumerate(time_ranges):
                while len(timestamps) - 1 > idx and (timestamps[idx] < time or idx <= time_idx):
                    idx += 1
                ranges.append(idx)
        except KeyError as err:
            self.logger.debug(err)
            self.logger.debug("Failed to sort by timestamps... only a reason to worry if this is BOHB-analysis")
            ranges = [int(x) for x in scale(1, runs_total, num=quantiles+1)]
        # Fix possible wrong values
        ranges[0] = 0
        ranges[-1] = len(as_list)

        self.logger.debug("Creating %d quantiles with a total number of runs of %d", quantiles, runs_total)
        self.logger.debug("Ranges: %s", str(ranges))

        for r in range(len(ranges))[1:]:
            if ranges[r] <= ranges[r-1]:
                if ranges[r-1] + 1 >= len(as_list):
                    raise RuntimeError("There was a problem with the quantiles of the configuration footprint. "
                                       "Please report this Error on \"https://github.com/automl/CAVE/issues\" and provide the debug.txt-file.")
                ranges[r] = ranges[r-1] + 1
                self.logger.debug("Fixed ranges to: %s", str(ranges))

        # Sanity check
        if not ranges[0] == 0 or not ranges[-1] == len(as_list) or not len(ranges) == quantiles + 1:
            raise RuntimeError("Sanity check on range-creation in configurator footprint went wrong. "
                               "Please report this Error on \"https://github.com/automl/CAVE/issues\" and provide the debug.txt-file.")

        tmp_rh = RunHistory(average_cost)
        for i, j in zip(ranges[:-1], ranges[1:]):
            for idx in range(i, j):
                k, v = as_list[idx]
                tmp_rh.add(config=rh.ids_config[k.config_id],
                           cost=v.cost, time=v.time, status=v.status,
                           instance_id=k.instance_id, seed=k.seed,
                           additional_info=v.additional_info)
            if timestamps:
                labels.append("{0:.2f}".format(timestamps[j - 1]))
            r_p_q_p_c.append([len(tmp_rh.get_runs_for_config(c)) for c in conf_list])
        self.logger.debug("Labels: " + str(labels))
        return labels, r_p_q_p_c

##################################################################################
##################################################################################
### PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING ###
##################################################################################
##################################################################################

    def _get_size(self, r_p_c):
        """Returns size of scattered points in dependency of runs per config

        Parameters
        ----------
        r_p_c: list[int]
            list with runs per config in order of self.conf_list

        Returns
        -------
        sizes: list[int]
            list with appropriate sizes for dots
        """
        normalization_factor = self.max_runs_per_conf - self.min_runs_per_conf
        min_size, enlargement_factor = 5, 20
        if normalization_factor == 0:  # All configurations same size
            normalization_factor = 1
            min_size = 12
        sizes = min_size + ((r_p_c - self.min_runs_per_conf) / normalization_factor) * enlargement_factor
        sizes *= np.array([0 if r == 0 else 1 for r in r_p_c])  # 0 size if 0 runs
        return sizes

    def _get_color(self, types):
        """Determine appropriate color for all configurations

        Parameters:
        -----------
        types: List[str]
            type of configuration

        Returns:
        --------
        colors: list
            list of color per config
        """
        colors = []
        for t in types:
            if t == "Default":
                colors.append('orange')
            elif "Incumbent" in t:
                colors.append('red')
            else:
                colors.append('white')
        return colors

    @timing
    def _plot_contour(self, p, contour_data, x_range, y_range):
        """Plot contour data.

        Parameters
        ----------
        p: bokeh.plotting.figure
            figure to be drawn upon
        contour_data: Dict[str -> np.array]
            dict from labels to array with contour data
        x_range: List[float, float]
            min and max of x-axis
        y_range: List[float, float]
            min and max of y-axis

        Returns
        -------
        handles: dict[str -> tuple(ImageGlyph, tuple(float, float))]
            mapping from label to image glyph and min/max-tuple
        """
        unique = np.unique(np.concatenate([contour_data[label][2] for label in contour_data.keys()]))
        color_mapper = LinearColorMapper(palette="Viridis256", low=np.min(unique), high=np.max(unique))
        handles = {}
        default_label = 'combined' if 'combined' in contour_data.keys() else list(contour_data.keys())[0]
        for label, data in contour_data.items():
            unique = np.unique(contour_data[label][2])
            handles[label] = (p.image(image=contour_data[label], x=x_range[0], y=y_range[0],
                                      dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0],
                                      color_mapper=color_mapper),
                              (np.min(unique), np.max(unique)))

            if not label == default_label and len(contour_data) > 1:
                handles[label][0].visible = False
        color_bar = ColorBar(color_mapper=color_mapper,
                             ticker=BasicTicker(desired_num_ticks=15),
                             label_standoff=12,
                             border_line_color=None, location=(0, 0))
        color_bar.major_label_text_font_size = '12pt'
        p.add_layout(color_bar, 'right')
        return handles, color_mapper

    def _create_views(self, source, used_configs):
        """Create views in order of plotting, so more interesting views are
        plotted on top. Order of interest:
        default > final-incumbent > incumbent > candidate
          local > random
            num_runs (ascending, more evaluated -> more interesting)
        Individual views are necessary, since bokeh can only plot one
        marker-type (circle, triangle, ...) per 'scatter'-call

        Parameters
        ----------:
        source: ColumnDataSource
            containing relevant information for plotting
        used_configs: List[Configuration]
            configs that are contained in this source. necessary to plot glyphs for the independent runs so they can be
            toggled. not all configs are in every source because of efficiency: no need to have 0-runs configs

        Returns
        -------
        views: List[CDSView]
            views in order of plotting
        views_by_run: Dict[ConfiguratorRun -> List[int]]
            maps each run to a list of indices of the related glyphs in the returned 'views'-list
        markers: List[string]
            markers (to the view with the same index)
        """

        def _get_marker(t, o):
            """ returns marker according to type t and origin o """
            if t == "Default":
                shape = 'triangle'
            elif t == 'Final Incumbent':
                shape = 'inverted_triangle'
            else:
                shape = 'square' if t == "Incumbent" else 'circle'
                shape += '_x' if o.startswith("Acquisition Function") else ''
            return shape

        views, markers = [], []
        views_by_run = {run : [] for run in self.configs_in_run}
        idx = 0
        for t in ['Candidate', 'Incumbent', 'Final Incumbent', 'Default']:
            for o in ['Unknown', 'Random', 'Acquisition Function']:
                for z in sorted(list(set(source.data['zorder'])), key=lambda x: int(x)):
                    for run, configs in self.configs_in_run.items():
                        booleans = [True if c in configs else False for c in used_configs]
                        view = CDSView(source=source, filters=[
                                GroupFilter(column_name='type', group=t),
                                GroupFilter(column_name='origin', group=o),
                                GroupFilter(column_name='zorder', group=z),
                                BooleanFilter(booleans)])
                        views.append(view)  # all views
                        views_by_run[run].append(idx)  # views sorted by runs
                        idx += 1
                        markers.append(_get_marker(t, o))
        self.logger.debug("%d different glyph renderers, %d different zorder-values",
                          len(views), len(set(source.data['zorder'])))
        return (views, views_by_run, markers)

    @timing
    def _scatter(self, p, source, views, markers):
        """
        Parameters
        ----------
        p: bokeh.plotting.figure
            figure
        source: ColumnDataSource
            data container
        views: List[CDSView]
            list with views to be plotted (in order!)
        markers: List[str]
            corresponding markers to the views

        Returns
        -------
        scatter_handles: List[GlyphRenderer]
            glyph renderer per view
        """
        scatter_handles = []
        for view, marker in zip(views, markers):
            scatter_handles.append(p.scatter(x='x', y='y',
                                             source=source,
                                             view=view,
                                             color='color', line_color='black',
                                             size='size',
                                             marker=marker,
                                             ))
        return scatter_handles

    def _plot_get_source(self,
                         conf_list,
                         runs,
                         X,
                         inc_list,
                         hp_names):
        """
        Create ColumnDataSource with all the necessary data
        Contains for each configuration evaluated on any run:

          - all parameters and values
          - origin (if conflicting, origin from best run counts)
          - type (default, incumbent or candidate)
          - # of runs
          - size
          - color

        Parameters
        ----------
        conf_list: list[Configuration]
            configurations
        runs: list[int]
            runs per configuration (same order as conf_list)
        X: np.array
            configuration-parameters as 2-dimensional array
        inc_list: list[Configuration]
            incumbents for this conf-run
        hp_names: list[str]
            names of hyperparameters

        Returns
        -------
        source: ColumnDataSource
            source with attributes as requested
        conf_list: List[Configuration]
            filtered conf_list with only configs we actually plot (i.e. > 0 runs)
        """
        # Remove all configurations without any runs
        keep = [i for i in range(len(runs)) if runs[i] > 0]
        runs = np.array(runs)[keep]
        conf_list = np.array(conf_list)[keep]
        X = X[keep]
        inc_list = [a for b in inc_list for a in b]

        source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1]))
        for k in hp_names:  # Add parameters for each config
            source.add([c[k] if c[k] else "None" for c in conf_list], escape_parameter_name(k))
        conf_types = ["Default" if c == self.default else "Final Incumbent" if c == self.final_incumbent
                      else "Incumbent" if c in inc_list else "Candidate" for c in conf_list]
        # We group "Local Search" and "Random Search (sorted)" both into local
        origins = [get_config_origin(c) for c in conf_list]
        source.add(conf_types, 'type')
        source.add(origins, 'origin')
        sizes = self._get_size(runs)
        sizes = [s * 3 if conf_types[idx] == "Default" else s for idx, s in enumerate(sizes)]
        source.add(sizes, 'size')
        source.add(self._get_color(source.data['type']), 'color')
        source.add(runs, 'runs')
        # To enforce zorder, we categorize all entries according to their size
        # Since we plot all different zorder-levels sequentially, we use a
        # manually defined level of influence
        num_bins = 20  # How fine-grained the size-ordering should be
        min_size, max_size = min(source.data['size']), max(source.data['size'])
        step_size = (max_size - min_size) / num_bins
        if step_size == 0:
            step_size = 1
        zorder = [str(int((s - min_size) / step_size)) for s in source.data['size']]
        source.add(zorder, 'zorder')  # string, so we can apply group filter

        return source, conf_list

    def plot(self,
             X,
             conf_list: list,
             runs_per_quantile,
             inc_list: list=None,
             contour_data=None,
             use_timeslider=False,
             use_checkbox=True,
             timeslider_labels=None):
        """
        plots sampled configuration in 2d-space;
        uses bokeh for interactive plot
        saves results in self.output, if set

        Parameters
        ----------
        X: np.array
            np.array with 2-d coordinates for each configuration
        conf_list: list
            list of ALL configurations in the same order as X
        runs_per_quantile: list[np.array]
            configurator-run to be analyzed, as a np.array with
            the number of target-algorithm-runs per config per quantile.
        inc_list: list
            list of incumbents (Configuration)
        contour_data: list
            contour data (xx,yy,Z)
        use_timeslider: bool
            whether or not to have a time_slider-widget on cfp-plot
            INCREASES FILE-SIZE DRAMATICALLY
        use_checkbox: bool
            have checkboxes to toggle individual runs

        Returns
        -------
        (script, div): str
            script and div of the bokeh-figure
        over_time_paths: List[str]
            list with paths to the different quantiled timesteps of the
            configurator run (for static evaluation)
        """
        if not inc_list:
            inc_list = []
        over_time_paths = []  # development of the search space over time

        hp_names = [k.name for k in  # Hyperparameter names
                    conf_list[0].configuration_space.get_hyperparameters()]

        # bokeh-figure
        x_range = [min(X[:, 0]) - 1, max(X[:, 0]) + 1]
        y_range = [min(X[:, 1]) - 1, max(X[:, 1]) + 1]

        # Get individual sources for quantiles
        sources, used_configs = zip(*[self._plot_get_source(conf_list, quantiled_run, X, inc_list, hp_names)
                                      for quantiled_run in runs_per_quantile])

        # We collect all glyphs in one list
        # Then we have to dicts to identify groups of glyphs (for interactivity)
        # They map the name of the group to a list of indices (of the respective glyphs that are in the group)
        # Those indices refer to the main list of all glyphs
        # This is necessary to enable interactivity for two inputs at the same time
        all_glyphs = []
        overtime_groups = {}
        run_groups = {run : [] for run in self.configs_in_run.keys()}

        # Iterate over quantiles (this updates overtime_groups)
        for idx, source, u_cfgs in zip(range(len(sources)), sources, used_configs):
            # Create new plot if necessary (only plot all quantiles in one single plot if timeslider is on)
            if not use_timeslider or idx == 0:
                p = self._create_figure(x_range, y_range)
                if contour_data is not None:  # TODO
                    contour_handles, color_mapper = self._plot_contour(p, contour_data, x_range, y_range)

            # Create views and scatter
            views, views_by_run, markers = self._create_views(source, u_cfgs)
            scatter_handles = self._scatter(p, source, views, markers)
            self.logger.debug("Quantile %d: %d scatter-handles", idx, len(scatter_handles))
            if len(scatter_handles) == 0:
                self.logger.debug("No configs in quantile %d (?!)", idx)
                continue

            # Add to groups
            start = len(all_glyphs)
            all_glyphs.extend(scatter_handles)
            overtime_groups[str(idx)] = [str(i) for i in range(start, len(all_glyphs))]
            for run, indices in views_by_run.items():
                run_groups[run].extend([str(start + i) for i in indices])

            # Write to file
            if self.output_dir:
                file_path = "cfp_over_time/configurator_footprint" + str(idx) + ".png"
                over_time_paths.append(os.path.join(self.output_dir, file_path))
                self.logger.debug("Saving plot to %s", over_time_paths[-1])
                export_bokeh(p, over_time_paths[-1], self.logger)

        # Add hovertool (define what appears in tooltips)
        # TODO add only important parameters (needs to change order of exec pimp before conf-footprints)
        hover = HoverTool(tooltips=[('type', '@type'), ('origin', '@origin'), ('runs', '@runs')] +
                                   [(k, '@' + escape_parameter_name(k)) for k in hp_names],
                          renderers=all_glyphs)
        p.add_tools(hover)

        # Build dashboard
        timeslider, checkbox, select_all, select_none, checkbox_title = self._get_widgets(all_glyphs, overtime_groups, run_groups,
                                                                                          slider_labels=timeslider_labels)
        contour_checkbox, contour_title = self._contour_radiobuttongroup(contour_handles, color_mapper)
        layout = p
        if use_timeslider:
            self.logger.debug("Adding timeslider")
            layout = column(layout, widgetbox(timeslider))
        if use_checkbox:
            self.logger.debug("Adding checkboxes")
            layout = row(layout,
                         column(widgetbox(checkbox_title),
                                widgetbox(checkbox),
                                row(widgetbox(select_all, width=100),
                                    widgetbox(select_none, width=100)),
                                widgetbox(contour_title),
                                widgetbox(contour_checkbox)))

        if self.output_dir:
            path = os.path.join(self.output_dir, "content/images/configurator_footprint.png")
            export_bokeh(p, path, self.logger)

        return layout, over_time_paths

    def _get_widgets(self, all_glyphs, overtime_groups, run_groups, slider_labels=None):
        """Combine timeslider for quantiles and checkboxes for individual runs in a single javascript-snippet

        Parameters
        ----------
        all_glyphs: List[Glyph]
            togglable bokeh-glyphs
        overtime_groups, run_groups: Dicŧ[str -> List[int]
            mapping labels to indices of the all_glyphs-list
        slider_labels: Union[None, List[str]]
            if provided, used as labels for timeslider-widget

        Returns
        -------
        time_slider, checkbox, select_all, select_none: Widget
            desired interlayed bokeh-widgets
        checkbox_title: Div
            text-element to "show title" of checkbox
        """
        aliases = ['glyph' + str(idx) for idx, _ in enumerate(all_glyphs)]
        labels_overtime = list(overtime_groups.keys())
        labels_runs = list(run_groups.keys())

        code = ""
        # Define javascript variable with important arrays
        code += "var glyphs = [" + ", ".join(aliases) + "];"
        code += "var overtime = [" + ','.join(['[' + ','.join(overtime_groups[l]) + ']' for l in labels_overtime]) + '];'
        code += "var runs = [" + ','.join(['[' + ','.join(run_groups[l]) + ']' for l in labels_runs]) + '];'
        # Deactivate all glyphs
        code += """
        glyphs.forEach(function(g) {
          g.visible = false;
        })"""
        # Add function for array-union (to combine all relevant glyphs for the different runs)
        code += """
        // union function
        function union_arrays(x, y) {
          var obj = {};
          for (var i = x.length-1; i >= 0; -- i)
             obj[x[i]] = x[i];
          for (var i = y.length-1; i >= 0; -- i)
             obj[y[i]] = y[i];
          var res = []
          for (var k in obj) {
            if (obj.hasOwnProperty(k))  // <-- optional
              res.push(obj[k]);
          }
          return res;
        }"""
        # Add logging
        code += """
        console.log("Timeslider: " + time_slider.value);
        console.log("Checkbox: " + checkbox.active);"""
        # Set timeslider title (to enable log-scale and print wallclocktime-labels)
        if slider_labels:
            code += "var slider_labels = " + str(slider_labels) + ";"
            code += "console.log(\"Detected slider_labels: \" + slider_labels);"
            code += "time_slider.title = \"Until wallclocktime \" + slider_labels[time_slider.value - 1] + \". Step no.\"; "
            title = "Until wallclocktime " + slider_labels[-1] + ". Step no. "
        else:
            title = "Quantile on {} scale".format("logarithmic" if self.timeslider_log else "linear")
            code += "time_slider.title = \"{}\";".format(title);
        # Combine checkbox-arrays, intersect with time_slider and set all selected glyphs to true
        code += """
        var activate = [];
        // if we want multiple checkboxes at the same time, we need to combine the arrays
        checkbox.active.forEach(function(c) {
          activate = union_arrays(activate, runs[c]);
        })
        // now the intersection of timeslider-activated and checkbox-activated
        activate = activate.filter(value => -1 !== overtime[time_slider.value - 1].indexOf(value));
        activate.forEach(function(idx) {
          glyphs[idx].visible = true;
        })
        """

        num_quantiles = len(overtime_groups)
        if num_quantiles > 1:
            timeslider = Slider(start=1, end=num_quantiles, value=num_quantiles, step=1, title=title)
        else:
            timeslider = Slider(start=1, end=2, value=1)
        labels_runs = [label.replace('_', ' ') if label.startswith('budget') else label for label in labels_runs]
        checkbox = CheckboxButtonGroup(labels=labels_runs, active=list(range(len(labels_runs))))

        args = {name: glyph for name, glyph in zip(aliases, all_glyphs)}
        args['time_slider'] = timeslider
        args['checkbox'] = checkbox
        callback = CustomJS(args=args, code=code)
        timeslider.js_on_change('value', callback)
        checkbox.callback = callback
        checkbox_title = Div(text="Showing only configurations evaluated in:")

        # Add all/none button to checkbox
        code_all  = "checkbox.active = " + str(list(range(len(labels_runs)))) + ";" + code
        code_none = "checkbox.active = [];" + code
        select_all  = Button(label="All", callback=CustomJS(args=args, code=code_all))
        select_none = Button(label="None", callback=CustomJS(args=args, code=code_none))

        return timeslider, checkbox, select_all, select_none, checkbox_title

    def _contour_radiobuttongroup(self, contour_data, color_mapper):
        """
        Returns
        -------
        radiobuttongroup: RadioButtonGroup
            radiobuttongroup widget to select one of the elements
        title: Div
            text-element to "show title" of widget
        """
        labels = [l.replace('_', ' ') if l.startswith('budget') else l for l in contour_data.keys()]
        aliases = ['glyph' + str(i) for i in range(len(labels))]
        values = list(contour_data.values())
        glyphs = [v[0] for v in values]
        mins = [v[1][0] for v in values]
        maxs = [v[1][1] for v in values]
        args = {name: glyph for name, glyph in zip(aliases, glyphs)}
        args['colormapper'] = color_mapper

        # Create javascript-code
        code = "var len_labels = " + str(len(aliases)) + ","
        code += "glyphs = [ " + ','.join(aliases) + '],'
        code += "mins = " + str(mins) + ','
        code += "maxs = " + str(maxs) + ';'

        code += """
            for (i = 0; i < len_labels; i++) {
                if (cb_obj.active === i) {
                    // console.log('Setting to true: ' + i);
                    glyphs[i].visible = true;
                    colormapper.low = mins[i];
                    colormapper.high = maxs[i];
                } else {
                    // console.log('Setting to false: ' + i);
                    glyphs[i].visible = false;
                }
            }
            """
        # Create the actual checkbox-widget
        callback = CustomJS(args=args, code=code)
        radio = RadioButtonGroup(labels=labels, active=0, callback=callback)
        title = Div(text="Data used to estimate contour-plot")
        return radio, title

    def _create_figure(self, x_range, y_range):
        p = figure(plot_height=500, plot_width=600,
                   tools=['save', 'box_zoom', 'wheel_zoom', 'reset'],
                   x_range=x_range, y_range=y_range)
        p.xaxis.axis_label = "MDS-X"
        p.yaxis.axis_label = "MDS-Y"
        p.xaxis.axis_label_text_font_size = "15pt"
        p.yaxis.axis_label_text_font_size = "15pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        p.title.text_font_size = "15pt"
        return p
