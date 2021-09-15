import numpy as np

class LogisticRegressor():
    def __init__(self, alpha=0.1, epochs=1, regularize=False, reg_factor=0.1):
        """
        Constructor

        alpha: learning rate
        epochs: the number of epochs (how many times does the LR see the whole dataset)
        regularize: whether or not the LR should implement regularization during training
        reg_factor: the regularization factor to use (lambda); only used if regularize is set to True
        """
        self.alpha = alpha
        self.epochs = epochs
        self.regularize = regularize
        self.reg_factor = reg_factor
        self.costs = []
        self.theta = None

    def _cost_function(self, hyp, y):
        """
        Calculates the cost function (error) for the predicted values (hyp) when compared against the right labels (y).
        hyp: the values predicted by the current configuration of the LR
        y: the correct labels

        TODO: Implement this function to calculate the cost value for the LR. 
        Recall that the LR might be set to perform regularization, so you must have two cases: one for when regularization is required and one for when it is not.
        This returns an scalar
        """
        m = len(y)
        #print(m)

        # Cost Function Without regularization
        # -1/m [Sum(y^i * log(h0(x^i)) + (1 - y^i) * log(1-h0(x^i)))]
        
        cost = -1/m * np.sum(y * np.log(hyp) + (1 - y) * np.log(1 - hyp))

        if self.regularize:
            #cost = -1/m * np.sum(y * np.log(hyp) + (1 - y) * np.log(1 - hyp)) + self.reg_factor/(2 * m) * np.sum(self.theta ** 2) 
            cost += [(self.reg_factor / (2 * m)) * np.sum(self.theta[1:]**2)] 
            # Cost Function With regularization
            # -1/m [Sum(y^i * log(h0(x^i)) + (1 - y^i) * log(1-h0(x^i)))] + (Lambda/2m) * Sum(Theta^2)

        return cost

    def _cost_function_derivative(self, y_pred, y, X, m):
        """
        Calculates the derivative (gradient) of the cost funcion. It supports the case for regularization.

        y_pred: the predicted values
        y: the target class (right values) for the data
        X: the input dataset
        m: the number of examples in the dataset

        TODO: implement this function to return an (n x 1) array that includes the derivative per feature (n is the num of features).
        Your implementation must support regularization, so you will have two cases here, one for when regularization is requested and another one for when it is not.

        """
        derivatives = np.dot((y_pred - y),X.T)
        empty_derivatives = (self.alpha/m) * derivatives.T 

        if self.regularize:
            derivatives = np.dot((y_pred - y),X.T)
            empty_derivatives = ((self.alpha/m) * derivatives.T) + np.sum((self.reg_factor / m) * self.theta)

        return empty_derivatives

    def _hypothesis(self, X):
        """
        Calculates the hypothesis for the given dataset using the current LR configuration (theta parameters). This is the sigmoid function.
        X: the dataset to employ. It is an (n x m) array.

        TODO: Implement this function to return a (1 x m) array with the list of predictions for each sample
        """
        
        #emptyResult = np.zeros((1, X.shape[1]))
        z = np.dot(self.theta.T,X)
        emptyResult = 1/(1+np.exp(-z))
        return emptyResult

    def fit(self, X, y):
        """
        Fits the Logistic Regressor to the values in the dataset
        X: is an (n x m) vector, where n is the number of features and m is the number of samples/examples
        y: is an (1 x m) vector, where m is the number of samples/examples

        TODO: You need to provide an implementation that in each epoch is updating the values for the theta parameters by using the hypothesis and cost function functions.
        """

        # m is the number of samples, n is the number of features, y is (1 x m)
        n, m = X.shape[0], X.shape[1]

        # self.theta is an (n x 1) vector (one per feature)
        self.theta = np.random.uniform(-10, 10, (n, 1))

        for i in range(self.epochs):
            # Get predictions
            # hyp = ...   # hyp is (1xm) vector
            hyp = self._hypothesis(X)

            # Calculate cost
            # cost = ...      # cost is a scalar
            cost = self._cost_function(hyp, y)

            # get gradient, an (nx1) array
            # gradient = ...
            gradient = self._cost_function_derivative(hyp, y, X, m)
            # delta/update rule
            # self.theta = ...
            self.theta = self.theta - gradient

            self.costs.append(cost)
            pass

        print("Final theta is {} (cost: {})".format(self.theta.T, cost))

    def predict(self, X):
        """
        Predicts the classes for the samples in the dataset X.

        X: an (n x m') array with the dataset to predict, m' samples of n dimensions.

        TODO: Implement this function to predict the class for the dataset X. You must return a (1 x m) array of 0|1. 
        The np.where function could be useful here to transform all outputs larger than 0.5 to 1.
        """
        #empty_predictions = np.zeros((1,X.shape[1]))
        empty_predictions = self._hypothesis(X)
        return np.where(empty_predictions < 0.5, 0, 1)
        
