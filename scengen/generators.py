import numpy as np
import pandas as pd

from scengen.vis.tree import VisTree


class PredClusGenerator:
    def __init__(self, sklearn_tree):
        self.tree = sklearn_tree

        self._clustering = None
        self._clustering_dict = None

    def to_visualization_tree(self, attribute_df, timeseries_df):
        return VisTree.from_sklearn_tree(self.tree.tree_, attribute_df, timeseries_df)
    @property
    def clustering(self):
        """
            Property that keeps track of the current clustering
        """
        return self._clustering

    @property
    def clustering_dict(self):
        """
            Property which is a dictionary from a leaf index to the indexes of the training data instances in particular leaf
            (This property is updated everytime the clustering property is set)
        """
        return self._clustering_dict

    @clustering.setter
    def clustering(self, clustering):
        self._clustering = clustering
        self._clustering_dict = {label: df.index.to_numpy() for label, df in self.clustering.groupby(self.clustering)}

    def fit(self, attributes: np.ndarray, timeseries: np.ndarray):
        # fit the tree
        self.tree.fit(attributes, timeseries)

        # record the 'clustering' of the training time series
        # dataframe to be able to use groupby
        self.clustering = pd.Series(self.tree.apply(attributes), dtype='int')

    def generate(self, attributes: np.ndarray, nb_of_samples: int):
        """
            Generates samples as a 2d numpy array
                samples[i,j] is the j'th sample for the i'th row in attributes
        """
        # let tree predict cluster membership
        predictions = self.tree.apply(attributes).astype('int')

        # for each prediction, take samples from the correct leaf
        cluster_dict = self.clustering_dict
        samples = np.zeros((attributes.shape[0], nb_of_samples), dtype='int')
        for i, prediction in enumerate(predictions):
            cluster = cluster_dict[prediction]
            samples[i, :] = np.random.choice(cluster, size=nb_of_samples, replace=True)

        return samples


class RandomGenerator:
    def __init__(self):
        self.nb_of_timeseries = None

    def fit(self, attributes, timeseries):
        self.nb_of_timeseries = timeseries.shape[0]

    def generate(self, attributes, nb_of_samples=250):
        return np.random.randint(0, self.nb_of_timeseries, size=(attributes.shape[0], nb_of_samples))


class SampleGenerator:
    """
        Usually the generators just return indices in the training data
        if you don't want to keep track of the training data yourself, this class wraps any generator producing indices
        and converts them to real data samples
    """

    def __init__(self, generator):
        self.generator = generator

        self.time_series = None

    def fit(self, attributes, timeseries):
        # fit the underlying generator
        self.generator.fit(attributes, timeseries)

        # save the training data
        self.time_series = timeseries

    def generate_timeseries(self, attributes, nb_of_samples=250):
        """
            generates time series from the underlying generator
            return value is an array (#test instances, #samples, #dimensions)
        """
        sample_indices = self.generator.generate(attributes, nb_of_samples)
        sample_timeseries = self.time_series[sample_indices]
        return sample_timeseries
