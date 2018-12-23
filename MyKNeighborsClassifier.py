#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.spatial import distance
import numpy as np
class MyKNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote similar to sklearn 
    library but different.
    https://goo.gl/Cmji3U

    But still same.
    
    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    method : string, optional (default = 'classical')
        method for voting. Possible values:
        - 'classical' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'weighted' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - 'validity' weights are calculated with distance and multiplied
          of validity for each voter.  
        Note: implementing kd_tree is bonus.
    norm : {'l1', 'l2'}, optional (default = 'l2')
        Distance norm. 'l1' is manhattan distance. 'l2' is euclidean distance.
    Examples
    --------
    """
    def __init__(self, n_neighbors=5, method='classical', norm='l2'):
        self.n_neighbors = n_neighbors
        self.method = method
        self.norm = norm
        self.labels = []
        self.distinct_labels = []
        self.data = []

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Training data. 
        y : array-like, shape = [n_samples] 
            Target values.
        """
        self.data = X
        self.labels = y
        self.distinct_labels = list(set(y))
        if(self.method == 'validity'):
            self.validity=[]

            for iterX in range(0,len(X)):
                x=X[iterX]
                if(self.norm == 'l2'):
                    temp = [distance.euclidean(x, data) for data in self.data]
                else:
                    temp = [distance.cityblock(x, data) for data in self.data]

                indices = np.asarray(temp).argsort()[:(self.n_neighbors+1)].astype(int).tolist()
                labels_of_test = np.asarray(self.labels)[indices]
                del temp[indices[0]]
                labels_of_test = labels_of_test.tolist()
                del labels_of_test[0]
                labels_of_test=np.asarray(labels_of_test)
                del indices[0]
                distances_to_indices = []
                if(self.norm == 'l2'):
                    for currentIndex in range(0, self.n_neighbors):
                        currentDistance = 1 / (distance.euclidean(x, self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                else:
                    for currentIndex in range(0, self.n_neighbors):
                        currentDistance = 1 / (distance.cityblock(x, self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                tempArray = []
                for i in range(0, len(self.distinct_labels)):
                    currentWeight = 0.0
                    currentDistinctLabel = self.distinct_labels[i]
                    indicesToGetWeights = np.where(labels_of_test == currentDistinctLabel)[0]
                    for itr in indicesToGetWeights:
                        currentWeight += distances_to_indices[itr]
                    tempArray.append(currentWeight)
                norm = np.linalg.norm(tempArray, ord=1)
                tempArray /= norm
                self.validity.append(tempArray[y[iterX]])

        pass
            
    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        if len(self.labels) == 0:
            raise ValueError("You should fit first!")
        resultProba = MyKNeighborsClassifier.predict_proba(self,X)
        returnArray = []
        for i in range(0,len(X)):
            returnArray.append(np.argmax(resultProba[i]))
        return returnArray

        pass
        
    def predict_proba(self, X, method=None):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features),
            Test samples.
        method : string, if None uses self.method.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        returnArray = []
        if(method == None):
            method = self.method
        if(method == 'classical'):

            if(self.norm == 'l2'):
                for x in X:
                    temp= [distance.euclidean(x, data) for data in self.data]
                    indices = np.asarray(temp).argsort()[:self.n_neighbors].astype(int).tolist()
                    labels_of_test = np.asarray(self.labels)[indices]
                    counts = np.bincount(labels_of_test)

                    tempArray= []
                    for i in range(0,len(self.distinct_labels)):
                        tempArray.append(float(counts[self.distinct_labels[i]])/self.n_neighbors)
                    returnArray.append(tempArray)
            if(self.norm == 'l1'):
                for x in X:
                    temp= [distance.cityblock(x, data) for data in self.data]
                    indices = np.asarray(temp).argsort()[:self.n_neighbors].astype(int).tolist()
                    labels_of_test = np.asarray(self.labels)[indices]
                    counts = np.bincount(labels_of_test)

                    tempArray= []
                    for i in range(0,len(self.distinct_labels)):
                        tempArray.append(float(counts[self.distinct_labels[i]])/self.n_neighbors)
                    returnArray.append(tempArray)

        if(method == 'weighted'):
            for x in X:
                if(self.norm == 'l2'):
                    temp = [distance.euclidean(x, data) for data in self.data]
                else:
                    temp = [distance.cityblock(x, data) for data in self.data]
                indices = np.asarray(temp).argsort()[:self.n_neighbors].astype(int).tolist()
                labels_of_test = np.asarray(self.labels)[indices]
                distances_to_indices = []
                if(self.norm == 'l2'):
                    for currentIndex in range(0,self.n_neighbors):
                        currentDistance = 1/(distance.euclidean(x,self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                else:
                    for currentIndex in range(0, self.n_neighbors):
                        currentDistance = 1 / (distance.cityblock(x, self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                tempArray = []
                for i in range(0, len(self.distinct_labels)):
                    currentWeight = 0.0
                    currentDistinctLabel = self.distinct_labels[i]
                    indicesToGetWeights = np.where(labels_of_test == currentDistinctLabel)[0]
                    for itr in indicesToGetWeights:
                        currentWeight += distances_to_indices[itr]
                    tempArray.append(currentWeight)
                norm = np.linalg.norm(tempArray, ord=1)
                tempArray/=norm
                returnArray.append(tempArray)

        if(method == 'validity'):

            for x in X:
                if (self.norm == 'l2'):
                    temp = [distance.euclidean(x, data) for data in self.data]
                else:
                    temp = [distance.cityblock(x, data) for data in self.data]
                indices = np.asarray(temp).argsort()[:self.n_neighbors].astype(int).tolist()
                labels_of_test = np.asarray(self.labels)[indices]
                distances_to_indices = []
                if(self.norm == 'l2'):
                    for currentIndex in range(0, self.n_neighbors):
                        currentDistance = self.validity[indices[currentIndex]] / (
                                distance.euclidean(x, self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                else:
                    for currentIndex in range(0, self.n_neighbors):
                        currentDistance = self.validity[indices[currentIndex]] / (
                                distance.cityblock(x, self.data[indices[currentIndex]]) + 1e-15)
                        distances_to_indices.append(currentDistance)
                tempArray = []
                for i in range(0, len(self.distinct_labels)):
                    currentWeight = 0.0
                    currentDistinctLabel = self.distinct_labels[i]
                    indicesToGetWeights = np.where(labels_of_test == currentDistinctLabel)[0]
                    for itr in indicesToGetWeights:
                        currentWeight += distances_to_indices[itr]
                    tempArray.append(currentWeight)
                norm = np.linalg.norm(tempArray, ord=1)
                tempArray /= norm
                returnArray.append(tempArray)

        return returnArray

        pass

if __name__=='__main__':
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = MyKNeighborsClassifier(n_neighbors=3,method="validity")
    neigh.fit(X, y)


    print(neigh.predict_proba([[0.9]], method='classical'))
    # [[0.66666667 0.33333333]]
    print(neigh.predict_proba([[0.9]], method='weighted'))
    # [[0.92436975 0.07563025]]
    print(neigh.predict_proba([[0.9]], method='validity'))
    # [[0.92682927 0.07317073]]
    print neigh.predict([[1.1]])