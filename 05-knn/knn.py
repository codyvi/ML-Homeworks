
import numpy as np
from collections import Counter


class Knn:
    def __init__(self, k):
        """
        Constructor
        """
        self.k = k
        self.model_x = None
        self.model_y = None

    def euclidean_distance(self, example1, example2):
        # TODO: Implement the euclidean distance function to calculate the distance between the provided elements
        # The output should be a scalar
        distance = 0.0
        distance = np.sqrt(np.sum((example1-example2.T)**2))
        return distance

    def get_neighbors(self, example):
        # TODO: Implement this function to obtain the k nearest neighbours for the provided example
        # You must return an np array of shape 1 x k, each element must be the index of the neighbour so you can associate it with its class/label later.
        # You will need to calculate the distance and then sort, recalling the indexes. Do your research to find a method that allows this.
        # Remember that in self.model_x you have your list of examples
        distances = [self.euclidean_distance(x, example) for x in self.model_x.T]
        k_indices = np.argsort(distances)[:self.k]
        k_indices = k_indices.reshape(1, self.k)
        return k_indices

    def fit(self, X, y):
        """
        The fit function, it only copies the entire dataset to class attributes.
        X is an np array of n x m shape, n is the number of features, m is the number of examples
        y is an np array of 1 x m shape, with the corresponding class for each example in X
        """
        self.model_x = X
        self.model_y = y

    def predict(self, example):
        # TODO: Implement this function to predict the class for the provided example.
        # example is an np.array of shape n x 1, where n is the number of features
        # You must return a scalar (int)
        # For each example, you will need to get the indexes of nearest neighbours and then the voting.
        # Do your research to find numpy or python functions that allow to sort and count
        # In self.model_y you will have the list of the classes for examples in original dataset (self.model_x). self.model_y is of shape 1xm
        x = self.get_neighbors(example)
        k_nearest_labels = [self.model_y[0][i] for i in x]
        most_common = Counter(k_nearest_labels[0]).most_common(1)
        #print(k_nearest_labels[0])
        #print(most_common)
        return most_common[0][0]
