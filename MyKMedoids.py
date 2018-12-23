#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance

class MyKMedoids:
    """KMedoids implementation parametric with 'pam' and 'clara' methods.

    Parameters
    ----------
    n_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of medoids to
        determine.
    max_iter : int, default: 300
        Maximum number of iterations of the k-medoids algorithm for a
        single run.
    method : string, default: 'pam'
        If it is pam, it applies pam algorithm to whole dataset 'pam'.
        If it is 'clara' it selects number of samples with sample_ratio and applies
            pam algorithm to the samples. Returns best medoids of all trials
            according to cost function.
    sample_ratio: float, default: .2
        It is used if the method is 'clara'
    clara_trials: int, default: 10,
        It is used if the method is 'clara'
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples

    """

    def __init__(self, n_clusters=3, max_iter=300, method='pam', sample_ratio=.2, clara_trials=10, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.method = method
        self.sample_ratio = sample_ratio
        self.clara_trials = clara_trials
        self.random_state = random_state
        self.best_medoids = []
        self.min_cost = float('inf')
        self.X=[]

    def fit(self, X):
        """Compute k-medoids clustering. If method is 'pam'
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
        Returns
        ----------
        self : MyKMedoids
        """

        # randomly initialize:
        self.random_state = np.random.RandomState(self.random_state)
        self.X = X
        if(self.method == 'pam'):
            self.best_medoids,min_cost = MyKMedoids.pam(self,X)
            cls = MyKMedoids.generate_clusters(self, self.best_medoids,X)
            self.min_cost = MyKMedoids.calculate_cost(self,self.best_medoids,cls)
        if(self.method == 'clara'):

            for i in range(0,self.clara_trials):
                currentSample = MyKMedoids.sample(self)
                current_min_cost_medoids, current_min_cost = MyKMedoids.pam(self, currentSample)
                clusters = MyKMedoids.generate_clusters(self,current_min_cost_medoids,X)
                cost_on_entire_dataset = MyKMedoids.calculate_cost(self, current_min_cost_medoids, clusters)
                print("Iteration "+str(i))
                print("selected samples: ")
                print(currentSample)
                print("sample best medoids: ")
                print(current_min_cost_medoids)
                print("sample min cost: "+str(current_min_cost))
                print("entire dataset cost: " + str(cost_on_entire_dataset))
                print("**************************************")
                if(cost_on_entire_dataset<self.min_cost):
                    self.min_cost = cost_on_entire_dataset
                    self.best_medoids = current_min_cost_medoids




        #self.best_medoids = np.asarray(medoids).astype(float)
        pass

    def sample(self):
        """Samples from the data with given sample_ratio.

        Returns
        -------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        """
        sampleIndices = self.random_state.choice(len(self.X), int(len(self.X)*self.sample_ratio), replace=False)

        return self.X[sampleIndices]
        pass

    def pam(self, X):
        """
        kMedoids - PAM
        See more : http://en.wikipedia.org/wiki/K-medoids
        The most common realisation of k-medoid clustering is the Partitioning Around Medoids (PAM) algorithm and is as follows:[2]
        1. Initialize: randomly select k of the n data points as the medoids
        2. Associate each data point to the closest medoid. ("closest" here is defined using any valid distance metric, most commonly Euclidean distance, Manhattan distance or Minkowski distance)
        3. For each medoid m
            For each non-medoid data point o
                Swap m and o and compute the total cost of the configuration
        4. Select the configuration with the lowest cost.
        5. repeat steps 2 to 4 until there is no change in the medoid.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        best_medoids, min_cost : tuple, shape [n_samples,]
            Best medoids found and the cost according to it.
        """

        #first step:
        initialMedoidsIndices = self.random_state.choice(len(X),self.n_clusters,replace=False)
        medoids=X[initialMedoidsIndices]
        clusters = MyKMedoids.generate_clusters(self, medoids, X)


        # delete medoid data from clusters for first appearence
        for i in range(0,len(clusters)):
            for j in range(0,len(clusters[i])):
                if np.all(clusters[i][j]==medoids[i]):
                    del clusters[i][j]
                    break

        min_cost=MyKMedoids.calculate_cost(self, medoids,clusters)
        min_cost_medoids = medoids.copy()
        return_min_cost =min_cost
        return_min_cost_medoids = medoids.copy()
        currentIter=0
        #print("initial centers in pam: " + str(medoids))
        #print("initial clusters in pam: " + str(clusters))
        #print("cost for initial clusters in pam: " + str(MyKMedoids.calculate_cost(self, medoids, clusters)))
        #2-5 steps:
        while(currentIter<self.max_iter):
            print("pam iteration " + str(currentIter))
            for m in range(0, len(medoids)):
                for o in range(0, len(clusters[m])):
                    if (clusters[m][o].tolist() in medoids.tolist()):
                        continue

                    temp=medoids[m].copy()
                    medoids[m] = clusters[m][o].copy()
                    clusters[m][o] = temp.copy()

                    current_cost = MyKMedoids.calculate_cost(self, medoids,clusters)
                    if(current_cost < min_cost):
                        min_cost=current_cost
                        min_cost_medoids=medoids.copy()
                    #print("after swap, temp medoids: " + str(medoids))
                    #print("clusters :" + str(clusters))
                    #print("temp cost: " + str(current_cost))
                    #print("**********")
                    temp = medoids[m].copy()
                    medoids[m] = clusters[m][o].copy()
                    clusters[m][o] = temp.copy()

            currentIter+=1
            #print("after iter "+str(currentIter)+ " medoids: "+str(min_cost_medoids)+" ##########")
            #print("clusters: "+str(clusters))
            if(return_min_cost > min_cost):

                return_min_cost_medoids = min_cost_medoids.copy()
                medoids = return_min_cost_medoids.copy()


                clusters = MyKMedoids.generate_clusters(self,medoids,X)
                # delete medoid data from clusters for first appearence
                for i in range(0, len(clusters)):
                    for j in range(0, len(clusters[i])):
                        if np.all(clusters[i][j] == medoids[i]):
                            del clusters[i][j]
                            break
                min_cost = MyKMedoids.calculate_cost(self,medoids,clusters)
                return_min_cost = min_cost

            else:
                break

        return (return_min_cost_medoids, return_min_cost)

        pass

    def generate_clusters(self, medoids, samples):
        """Generates clusters according to distance to medoids. Order
        is same with given medoids array.
        Parameters
        ----------
        medoids: array_like, shape = [n_clusters, n_features]
        samples: array-like, shape = [n_samples, n_features]
        Returns
        -------
        clusters : array-like, shape = [n_clusters, elemens_inside_cluster, n_features]
        """
        clusters = []
        for i in range(0, medoids.shape[0]):
            clusters.append([])
        for currentSampleIndex in range(0, samples.shape[0]):
            currentSample = samples[currentSampleIndex]
            minDistance = np.inf
            minDistanceIndex = 0
            for currentMedoidIndex in range(0, medoids.shape[0]):
                currentDistance = distance.euclidean(currentSample, medoids[currentMedoidIndex])
                if (currentDistance < minDistance):
                    minDistance = currentDistance
                    minDistanceIndex = currentMedoidIndex
            clusters[minDistanceIndex].append(currentSample)
        return clusters
        pass

    def calculate_cost(self, medoids, clusters):
        """Calculates cost of each medoid's cluster with squared euclidean function.
        Parameters
        ----------
        medoids: array_like, shape = [n_clusters, n_features]
        clusters: array-like, shape = [n_clusters, elemens_inside_cluster, n_features]
        Returns
        -------
        cost : float
            total cost of clusters
        """
        cost = 0.0
        for i in range(0, len(medoids)):
            for j in range(0, len(clusters[i])):
                cost += distance.sqeuclidean(medoids[i], clusters[i][j])
        return cost
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
        labels = []
        for i in range(0,len(X)):
            min_distance = distance.euclidean(X[i],self.best_medoids[0])
            min_distance_index = 0

            for j in range(1,len(self.best_medoids)):
                current_distance = distance.euclidean(X[i],self.best_medoids[j])
                if(current_distance < min_distance):
                    min_distance = current_distance
                    min_distance_index = j

            labels.append(min_distance_index)
        return labels

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
        MyKMedoids.fit(self, X)
        return MyKMedoids.predict(self, X)
        pass


if __name__ == "__main__":
    X = np.array([np.array([2., 6.]),
                  np.array([3., 4.]),
                  np.array([3., 8.]),
                  np.array([4., 7.]),
                  np.array([6., 2.]),
                  np.array([6., 4.]),
                  np.array([7., 3.]),
                  np.array([7., 4.]),
                  np.array([8., 5.]),
                  np.array([7., 6.])

                  ])


    #kmedoids = MyKMedoids(n_clusters=2, random_state=0)
    kmedoids = MyKMedoids(method='pam', n_clusters=2, sample_ratio=1, clara_trials=10, max_iter=300, random_state=0)
    print kmedoids.fit_predict(X)
    # [1 1 1 1 0 0 0 0 0 0]
    print kmedoids.best_medoids
    # [array([7., 4.]), array([2., 6.])]
    print kmedoids.min_cost
    # 28.0
