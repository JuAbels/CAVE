import logging
import os

import numpy as np
from numpy import corrcoef

from scipy.cluster.hierarchy import linkage
from scipy.misc import comb

from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
from sklearn.tree.tests.test_tree import y_random
import matplotlib.pyplot as plt

from plottingscripts.plotting.scatter import plot_scatter_plot

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "MIT"
__email__ = "lindauer@cs.uni-freiburg.de"


class FeatureAnalysis(object):

    def __init__(self,
                 output_dn: str,
                 scenario,
                 feat_names):
        '''
        From: https://github.com/mlindauer/asapy


        Arguments
        ---------
        output_dn: str
            output directory name
        scenario: Scenario
            scenario for features
        '''
        self.logger = logging.getLogger("Feature Analysis")
        self.scenario = scenario
        self.feat_names = scenario.feature_names
        self.feature_data = {}
        for name in feat_names:
            insts = self.scenario.train_insts
            insts.extend(self.scenario.test_insts)
            self.feature_data[name] = {}
            for i in insts:
                self.feature_data[name][i] = self.scenario.feature_dict[i][feat_names.index(name)]
        self.feature_data = DataFrame(self.feature_data)

        self.output_dn = os.path.join(output_dn, "feature_plots")
        if not os.path.isdir(self.output_dn):
            os.makedirs(self.output_dn)

    def get_box_violin_plots(self):
        '''
            for each feature generate a plot with box and vilion plot

            Parameters
            ----------
            feat_names: list[str]
                names of the features

            Returns
            -------
            list of tuples of feature name and feature plot file name
        '''
        self.logger.info("Plotting box and violin plots........")

        files_ = []

        for feat_name in sorted(self.feat_names):
            matplotlib.pyplot.close()
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
            vec = self.feature_data[feat_name].values
            vec = vec[~np.isnan(vec)]
            axes[0].violinplot(
                [vec], showmeans=False, showmedians=True, vert=False)
            axes[0].xaxis.grid(True)
            plt.setp(axes[0], yticks=[1], yticklabels=[""])
            axes[1].boxplot(vec, vert=False)
            axes[1].xaxis.grid(True)
            plt.setp(axes[1], yticks=[1], yticklabels=[""])

            plt.tight_layout()

            out_fn = os.path.join(
                self.output_dn, "violin_box_%s_plot.png" % (feat_name.replace("/", "_")))
            plt.savefig(out_fn)
            files_.append((feat_name, out_fn))

        return files_

    def correlation_plot(self):
        '''
            generate correlation plot using spearman correlation coefficient and ward clustering
            Returns
            -------
            file name of saved plot
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting correlation plots........")

        feature_data = self.feature_data
        features = list(self.feature_data.columns)
        feature_data = feature_data.fillna(feature_data.mean())
        feature_data = feature_data.values

        n_features = len(features)

        data = np.zeros((n_features, n_features)) + 1  # similarity
        for i in range(n_features):
            for j in range(i + 1, n_features):
                rho = corrcoef([feature_data[:, i], feature_data[:, j]])[0, 1]
                if np.isnan(rho):  # is nan if one feature vec is constant
                    rho = 0
                data[i, j] = rho
                data[j, i] = rho

        link = linkage(data * -1, 'ward')  # input is distance -> * -1

        sorted_features = [[a] for a in features]
        for l in link:
            new_cluster = sorted_features[int(l[0])][:]
            new_cluster.extend(sorted_features[int(l[1])][:])
            sorted_features.append(new_cluster)

        sorted_features = sorted_features[-1]

        # resort data
        indx_list = []
        for f in features:
            indx_list.append(sorted_features.index(f))
        indx_list = np.argsort(indx_list)
        data = data[indx_list, :]
        data = data[:, indx_list]

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

        plt.xlim(0, data.shape[0])
        plt.ylim(0, data.shape[0])

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(sorted_features, minor=False)
        ax.set_yticklabels(sorted_features, minor=False)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=45, fontsize=2, ha="left")
        labels = ax.get_yticklabels()
        plt.setp(labels, rotation=0, fontsize=2, ha="right")

        fig.colorbar(heatmap)

        plt.tight_layout()

        out_plot = os.path.join(
            self.output_dn, "correlation_plot_features.png")
        plt.savefig(out_plot, format="png", dpi=400)

        return out_plot

    def cluster_instances(self):
        '''
            use pca to reduce feature dimensions to 2 and cluster instances using k-means afterwards
        '''
        matplotlib.pyplot.close()
        self.logger.info("Plotting clusters........")
        # impute missing data; probably already done, but to be on the safe
        # side
        self.feature_data = self.feature_data.fillna(
            self.feature_data.mean())

        # feature data
        features = self.feature_data.values

        # scale features
        ss = StandardScaler()
        features = ss.fit_transform(features)

        # feature reduction: pca
        pca = PCA(n_components=2)
        features = pca.fit_transform(features)

        # cluster with k-means
        scores = []
        for n_clusters in range(2, 12):
            km = KMeans(n_clusters=n_clusters)
            y_pred = km.fit_predict(features)
            score = silhouette_score(features, y_pred)
            scores.append(score)

        best_score = max(scores)
        best_run = scores.index(best_score)
        n_clusters = best_run + 2
        km = KMeans(n_clusters=n_clusters)
        y_pred = km.fit_predict(features)

        plt.figure()
        plt.scatter(features[:, 0], features[:, 1], c=y_pred)

        plt.tight_layout()
        out_fn = os.path.join(self.output_dn, "feature_clusters.png")
        plt.savefig(out_fn, format="png")

        return out_fn