#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance
EPSILON = 0.0001


class MyKMeans:
    """K-Means clustering similar to sklearn 
    library but different.
    https://goo.gl/bnuM33

    But still same.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_method : string, optional, default: 'random'
        Initialization method. Values can be 'random', 'kmeans++'
        or 'manual'. If 'manual' then cluster_centers need to be set.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    cluster_centers : np.array, used only if init_method is 'manual'.
        If init_method is 'manual' without fitting, these values can be used
        for prediction.
    """

    def __init__(self, init_method="random", n_clusters=3, max_iter=300, random_state=None, cluster_centers=[]):
        self.init_method = init_method
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        if init_method == "manual":
            self.cluster_centers = cluster_centers
        else:
            self.cluster_centers = []

    def fit(self, X):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self : MyKMeans
        """
        #MyKMeans.initialize(self, X)
        numberOfSamples = X.shape[0]
        currentIteration = 0

        while (currentIteration < self.max_iter):
            currentIteration += 1
            clusters = []
            for _ in range(0, self.n_clusters):
                clusters.append([])

            # find clusters according to centers:
            for i in range(0, numberOfSamples):
                min_distance = np.inf
                min_distance_index = -1
                for j in range(0, self.n_clusters):
                    current_distance = np.linalg.norm(X[i] - self.cluster_centers[j])
                    if (min_distance > current_distance):
                        min_distance = current_distance
                        min_distance_index = j
                clusters[min_distance_index].append(X[i])

            newClusterCenters = np.zeros((self.n_clusters, X.shape[1]))
            clusters = np.asarray(clusters)

            # find new centers according to clusters:
            for currentClusterIndex in range(0, self.n_clusters):
                if(clusters[currentClusterIndex] == []):
                    newClusterCenters[currentClusterIndex]= self.cluster_centers[currentClusterIndex]
                else:
                    newClusterCenters[currentClusterIndex] = np.asarray(clusters[currentClusterIndex]).mean(axis=0)

            if (np.linalg.norm(self.cluster_centers - newClusterCenters) <= EPSILON): break

            self.cluster_centers = newClusterCenters.copy()

        self.labels = np.zeros((numberOfSamples), int)
        self.labels = MyKMeans.predict(self, X)

        return self
        pass

    def initialize(self, X):
        """ Initialize centroids according to self.init_method
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self.cluster_centers : array-like, shape=(n_clusters, n_features)
        """
        numberOfSamples = X.shape[0]
        numberOfFeatures = X.shape[1]

        if self.init_method == 'random':
            self.cluster_centers = np.zeros((self.n_clusters, numberOfFeatures))
            #centers = self.random_state.permutation(np.arange(numberOfSamples))[:self.n_clusters]
            centers = self.random_state.choice(numberOfSamples,self.n_clusters,replace=False)
            self.cluster_centers = X[centers]
            self.cluster_centers = self.cluster_centers.astype(float)

        if self.init_method == 'kmeans++':
            firstCenterIndex = self.random_state.randint(numberOfSamples)
            X = X.astype('float').tolist()
            clusterCentersList = []
            clusterCentersList.append(X[firstCenterIndex])

            for i in range(1, self.n_clusters):
                max_distance = 0
                max_distance_index = 0

                for j in range(0, numberOfSamples):
                    if X[j] not in clusterCentersList:
                        current_distance = 0

                        for k in range(0, i):
                            current_distance += np.linalg.norm(np.asarray(X[j]) - np.asarray(clusterCentersList[k]))

                        if (current_distance > max_distance):
                            max_distance = current_distance
                            max_distance_index = j

                clusterCentersList.append(X[max_distance_index])

            self.cluster_centers = np.asarray(clusterCentersList)
        X=np.asarray(X)
        return self.cluster_centers
        pass

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        predictResult = []
        for predictPoint in X:
            min_distance = np.linalg.norm(np.asarray(predictPoint - self.cluster_centers[0]))
            min_distance_index = 0
            for i in range(1, self.n_clusters):
                current_distance = np.linalg.norm(np.asarray(predictPoint - self.cluster_centers[i]))
                if (current_distance < min_distance):
                    min_distance = current_distance
                    min_distance_index = i
            predictResult.append(min_distance_index)
        predictResult = np.asarray(predictResult)
        return predictResult
        pass

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        MyKMeans.fit(self, X)
        return MyKMeans.predict(self, X)
        pass


if __name__ == "__main__":
    if __name__ == "__main__":
        X = np.array([[1, 3, 5], [2, 5, 8], [1, 3, 7], [4, 2, 3], [7, 6, 3], [50, 60, 70], [50, 60, 70]])
        X = np.array([[1, 3, 5], [1, 3, 6], [1, 3, 5], [1, 3, 5], [1, 3, 5],[50, 60, 70]])
        X = np.array(
            [[1, 3, 5], [2, 5, 8], [1, 3, 7], [4, 2, 3], [50, 60, 70], [50, 60, 70], [50, 60, 70], [50, 60, 70],
             [50, 60, 70], [50, 60, 70]])
        kmeans = MyKMeans(n_clusters=5, init_method='kmeans++', random_state=0)
        print kmeans.initialize(X)
        kmeans.fit(X)
        print kmeans.cluster_centers
        print kmeans.predict(X)

