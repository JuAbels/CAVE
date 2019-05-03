import unittest
import numpy as np
import os
# print(os.getcwd())
from smac.runhistory.runhistory import RunHistory, DataOrigin
from sklearn.metrics import euclidean_distances
from cave.reader.configurator_run import ConfiguratorRun

from cave.plot.mds import MDS_BA
from cave.plot.configurator_footprint import average_cost
import cave.utils.helpers as help


def calculate_costvalue(dists, red_dists):
    """Only for testing"""
    low_dists = euclidean_distances(red_dists)
    n_conf = dists.shape[0]
    costvalue = []
    for i in range(n_conf - 1):
        for j in range(i + 1, n_conf):
            costvalue.append((dists[i][j] - low_dists[i][j]) ** 2)
    costvalue = sum(costvalue)
    return costvalue


runs = [(ConfiguratorRun('../../Branin/Smac3/smac3-output/run_1/',
                         '../../Branin/Smac3/',
                         file_format='SMAC3',
                         validation_format='NONE'))]

global_original_rh = RunHistory(average_cost)
global_validated_rh = RunHistory(average_cost)
global_epm_rh = RunHistory(average_cost)

for run in runs:
    global_original_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
    global_validated_rh.update(run.original_runhistory, origin=DataOrigin.INTERNAL)
    if run.validated_runhistory:
        global_validated_rh.update(run.validated_runhistory, origin=DataOrigin.EXTERNAL_SAME_INSTANCES)

global_epm_rh.update(global_validated_rh)
runs = sorted(runs, key=lambda run: global_epm_rh.get_cost(run.solver.incumbent))


class TestRunhistory(unittest.TestCase):

    def test_classification(self):
        """Function to test, if random and local runhistory created correctly"""
        # combined = help.combine_runhistories(runs)
        random, local = help.create_random_runhistories(global_original_rh)
        random = str(random.data)
        result = "OrderedDict([(RunKey(config_id=1, instance_id=None, seed=0), RunValue(cost=15.336866, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=2, instance_id=None, seed=0), RunValue(cost=6.042356, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=3, instance_id=None, seed=0), RunValue(cost=76.191931, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=4, instance_id=None, seed=0), RunValue(cost=2.102303, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=5, instance_id=None, seed=0), RunValue(cost=11.997325, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=6, instance_id=None, seed=0), RunValue(cost=27.718052, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=7, instance_id=None, seed=0), RunValue(cost=123.166119, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=8, instance_id=None, seed=0), RunValue(cost=13.556812, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=9, instance_id=None, seed=0), RunValue(cost=3.265477, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=10, instance_id=None, seed=0), RunValue(cost=20.514643, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=11, instance_id=None, seed=0), RunValue(cost=2.522322, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=12, instance_id=None, seed=0), RunValue(cost=20.305772, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=13, instance_id=None, seed=0), RunValue(cost=11.241301, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=14, instance_id=None, seed=0), RunValue(cost=91.710919, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=15, instance_id=None, seed=0), RunValue(cost=20.195214, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=16, instance_id=None, seed=0), RunValue(cost=188.443026, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=17, instance_id=None, seed=0), RunValue(cost=82.295484, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=18, instance_id=None, seed=0), RunValue(cost=31.528635, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=19, instance_id=None, seed=0), RunValue(cost=77.229825, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=20, instance_id=None, seed=0), RunValue(cost=97.872458, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=21, instance_id=None, seed=0), RunValue(cost=140.605172, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=22, instance_id=None, seed=0), RunValue(cost=18.810533, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=23, instance_id=None, seed=0), RunValue(cost=0.933445, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=24, instance_id=None, seed=0), RunValue(cost=14.273374, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={})), (RunKey(config_id=25, instance_id=None, seed=0), RunValue(cost=37.695976, time=-1.0, status=<StatusType.SUCCESS: 1>, additional_info={}))])"
        # self.assertEqual(random, result)

    def test_combination(self):
        """Test if the right combined runhistory is created using random and local runhistory"""
        # combine_random_local
        pass

    def test_list_rhs(self):
        """Test if the combined runhistorys are listed correctly."""
        # create_new_rhs
        pass


class TestMdsMethods(unittest.TestCase):

    def test_fit(self):
        """Function to test, if the fitting of the classical method works correctly"""
        os.chdir(os.path.dirname(__file__))
        print("TEST")
        print(os.getcwd())
        mds = MDS_BA(n_components=2, dissimilarity="precomputed", method='inductive')
        matrix = np.array([[0,   93,  82, 133],
                           [93,   0,  52,  60],
                           [82,  52,   0, 111],
                           [133, 60, 111,   0]])
        mds.fit(matrix)

        eigenvalues = np.array([9724.168, 3160.986])
        eigenvectors = np.array([[-0.6370,  0.586],
                                 [0.187,   -0.214],
                                 [-0.2530, -0.706],
                                 [0.704,    0.334]])
        np.testing.assert_allclose(eigenvalues, mds.e_vals_[:mds.n_components])
        np.testing.assert_allclose(eigenvectors, mds.e_vecs_[:, :mds.n_components], rtol=1e-2)

    def test_embedding(self):
        """
        Function to test, if the embedding space of the random configurations is
        calculated correctly.
        """
        mds = MDS_BA(n_components=2, dissimilarity="precomputed", method='inductive')
        matrix = np.array([[0, 93, 82, 133],
                           [93, 0, 52, 60],
                           [82, 52, 0, 111],
                           [133, 60, 111, 0]])
        mds.fit(matrix)
        results = mds.embedding(mds.n_components)
        embedding = np.array([[-62.831,  32.97448],
                              [ 18.403, -12.02697],
                              [-24.960, -39.71091],
                              [ 69.388,  18.76340]])
        np.testing.assert_allclose(embedding, results, rtol=1e-04)

    def test_transfomation_one(self):
        """Test, if extending of new points in embedding space works"""
        mds = MDS_BA(n_components=2, dissimilarity="precomputed", method='inductive')
        matrix = np.array([[0, 93, 82, 133],
                           [93, 0, 52, 60],
                           [82, 52, 0, 111],
                           [133, 60, 111, 0]])
        mds.fit(matrix)
        result = mds.embedding(mds.n_components)
        kernel = mds.center_similarities(matrix, matrix)
        new_embedding = mds.transform(kernel)

        np.testing.assert_allclose(result, new_embedding)

    def test_transformation_two(self):
        sim = np.array([[0, 5, 3, 4],
                        [5, 0, 2, 2],
                        [3, 2, 0, 1],
                        [4, 2, 1, 0]])

        mds_clf = MDS_BA(dissimilarity="euclidean", method="inductive")
        mds_clf.fit(sim)

        # Testing for extending MDS to new points
        sim2 = np.array([[3, 1, 1, 2],
                         [4, 1, 2, 2]])
        result = mds_clf.transform(sim2)
        expected = np.array([[0.75683416, 0.37837926],
                             [1.14737093, 1.24472724]])
        np.testing.assert_allclose(result, expected)

    def test_distance_difference(self):
        """Test if the difference of original and embedding space calculated correclty"""
        original = np.array([[0, 93, 82, 133],
                             [93, 0, 52, 60],
                             [82, 52, 0, 111],
                             [133, 60, 111, 0]])
        new_coordinates = np.array([[-62.831,  32.97448],
                                    [ 18.403, -12.02697],
                                    [-24.960, -39.71091],
                                    [ 69.388,  18.76340]])

        result = calculate_costvalue(original, new_coordinates)

        np.isclose(result, 0.518401)

