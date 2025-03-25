import numpy as np
import random as random

class optimizer:
    '''
    A class of algorithms to solve optimization problem
    ...
    Attributes
    ----------
    model : str
        a string that will be printed in report.
    feature : dataframe
        store pd.DataFrame for independent variable
    numFeature : int
        an integer specifying length of feature
    featureMean : list
        a list of mean calculated for each feature
    featureSD : list
        a list of std. deviation calculated for each feature
    y : list
        a list of dependent variable
    epoch : int
        an integer storing number of completed epoch needed to train before converge
    coefficient : list
        a list of optimized coefficient for each feature
    cost_history : list
        a list storing cost/loss for each epoch
        
    Methods
    -------
    batchGD(intercept = 0, epoch = 1000, learning_rate = 0.01, threshold = 0)
        Runs a batch gradient descent algorithm
    
    stochasticGD(intercept = 0, epoch = 1000, learning_rate = 0.01, threshold = 0)
        Runs a stochastic gradient descent algorithm
    
    cost_function(coef, X, y)
        A static method to compute cost/loss value of model
        
    gradient_grad_desc(coef, X, y)
        A static method to compute gradient of the cost/loss function
    
    normalized(feature)
        A static method to normalized feature
    '''
    def __init__(self, feature, y, data, model = 'Linear Regression', normalized = 0):
        '''        
        Parameters
        ----------
        feature : list
            a list of column name for independent variable
        y : str
            a string specifying column to predict (dependent variable)
        data : dataframe
            a pd.DataFrame to train model
        model : str, optional
            a string to be printed in report (Default : `Linear Regression`)
        normalized : bool, optional
            a boolean to specify if need to normalize feature before training (Default = False)
        '''
        self.model = model
        self.numFeature = len(feature)
        self.y = data[y]
        if normalized:
            self.feature, self.featureMean, self.featureSD = self.normalized(data.loc[:,feature])
        else:
            self.feature = data.loc[:,feature]
        
    def __str__(self):
        print("Model Optimized: %s" % (self.model))
        print("Dataframe: %i Feature(s), %i Observation(s)" % (self.numFeature, len(self.y)))
        print("Epoch: %i" % (self.epoch))
        print("Estimated Coefficient:\n", self.coefficient)
        print()
        return 'training end...'
    
    def batchGD(self, intercept = 0, epoch = 1000, learning_rate = 0.01, threshold = 0):
        '''
        Method to run batch gradient descent for `optimizer` object. i.e: obj.batchGD()
        
        Parameters
        ----------
        intercept : bool, optional
            a boolean specifying if intercept need to be included. (Default: False)
        epoch : int, optional
            an int to specify iteration of one complete training. (Default: 1000) 
        learning_rate : int, optional
            an int to specify step size of descent (Default: 0.01)
        threshold : int, optional
            an int specifying difference between current and previous cost/loss value. Used as a stop rule (Default = 0)
        '''
        self.coefficient = np.random.randn(1,self.numFeature)[0]
        if intercept:
            self.coefficient = np.append(self.coefficient, np.random.randn(1,1)[0])
            self.feature = self.feature.assign(intercept = np.ones(len(self.y)))
        iter = 1
        difference = 1
        self.cost_history = [1]
        while(difference > threshold and iter < epoch):
            gradient = self.gradient_grad_desc(self.coefficient, self.feature, self.y)
            self.coefficient -= learning_rate * gradient
            cost_function = self.cost_function(self.coefficient, self.feature, self.y)
            self.cost_history.append(cost_function)
            difference = abs(cost_function - self.cost_history[iter-1])
            iter += 1
        self.cost_history.pop(0)
        self.epoch = iter 
        
    def stochasticGD(self, intercept = 0, epoch = 1000, learning_rate = 0.01, threshold = 0):
        '''
        Method to run stochastic gradient descent for `optimizer` object. i.e: obj.stochasticGD()
                
        Parameters
        ----------
        intercept : bool, optional
            a boolean specifying if intercept need to be included. (Default: False)
        epoch : int, optional
            an int to specify iteration of one complete training. (Default: 1000) 
        learning_rate : int, optional
            an int to specify step size of descent (Default: 0.01)
        threshold : int, optional
            an int specifying difference between current and previous cost/loss value. Used as a stop rule (Default = 0)
        '''
        self.coefficient = np.random.randn(1,self.numFeature)[0]
        if intercept:
            self.coefficient = np.append(self.coefficient, np.random.randn(1,1)[0])
            self.feature = self.feature.assign(intercept = np.ones(len(self.y)))
        iter = 1
        difference = 1
        self.cost_history = [1]
        while(difference > threshold and iter < epoch):
            random_index = random.choice(range(0, len(self.y) - 1))
            new_Feature, new_y = self.feature.loc[[random_index]], self.y.loc[[random_index]]
            gradient = self.gradient_grad_desc(self.coefficient, new_Feature, new_y)
            self.coefficient -= learning_rate * gradient
            cost_function = self.cost_function(self.coefficient, new_Feature, new_y)
            self.cost_history.append(cost_function)
            difference = abs(cost_function - self.cost_history[iter-1])
            iter += 1
        self.cost_history.pop(0)
        self.epoch = iter 
    
    def adaGrad(self, intercept = 0, epoch = 1000, learning_rate = 0.01, threshold = 0):
        '''TO-DO'''
        self.coefficient = np.random.randn(1,self.numFeature)[0]
        if intercept:
            self.coefficient = np.append(self.coefficient, np.random.randn(1,1)[0])
            self.feature = self.feature.assign(intercept = np.ones(len(self.y)))
        iter = 1
        difference = 1
        self.cost_history = [1]
        gradient_history = []
        outer_grad = []
        while(difference > threshold and iter < epoch):
            random_index = random.choice(range(0, len(self.y) - 1))
            new_Feature, new_y = self.feature.loc[[random_index]], self.y.loc[[random_index]]
            gradient = self.gradient_grad_desc(self.coefficient, new_Feature, new_y)
            gradient_history.append(gradient)
            outer_grad.append(gradient**2)
            diag_G = np.sum(outer_grad, axis = 0)
            stepsize = learning_rate / np.sqrt(1e-6 + diag_G)
            self.coefficient -= np.multiply(stepsize,gradient)
            cost_function = self.cost_function(self.coefficient, self.feature, self.y)
            self.cost_history.append(cost_function)
            difference = abs(cost_function - self.cost_history[iter-1])
            iter += 1
        self.cost_history.pop(0)
        self.epoch = iter 
    
    def adaDelta(self, intercept = 0, epoch = 1000, threshold = 0, decay = 0.95):
        '''TO-DO'''
        self.coefficient = np.random.randn(1,self.numFeature)[0]
        if intercept:
            self.coefficient = np.append(self.coefficient, np.random.randn(1,1)[0])
            self.feature = self.feature.assign(intercept = np.ones(len(self.y)))
        iter = 1
        difference = 1
        self.cost_history = [1]
        accumulate_grad = np.zeros(len(self.coefficient))
        accumulate_stepsize = np.zeros(len(self.coefficient))
        while(difference > threshold and iter < epoch):
            random_index = random.choice(range(0, len(self.y) - 1))
            new_Feature, new_y = self.feature.loc[[random_index]], self.y.loc[[random_index]]
            gradient = self.gradient_grad_desc(self.coefficient, new_Feature, new_y)
            accumulate_grad = accumulate_grad * decay + (1-decay) * (gradient**2)
            rms_grad = np.sqrt(accumulate_grad + 1e-6)
            rms_stepsize = np.sqrt(accumulate_stepsize + 1e-6)
            stepsize = -rms_stepsize / rms_grad * gradient
            accumulate_stepsize = accumulate_stepsize * decay + (1-decay) * (stepsize**2)
            self.coefficient += stepsize
            cost_function = self.cost_function(self.coefficient, self.feature, self.y)
            self.cost_history.append(cost_function)
            difference = abs(cost_function - self.cost_history[iter-1])
            iter += 1
        self.cost_history.pop(0)
        self.epoch = iter
    
    def rmsProp(self):
        '''TO-DO'''
        return 0
    
    def adaM(self):
        '''TO-DO'''
        return 0
    
    @staticmethod 
    def cost_function(coef, X, y):
        '''
        Method to compute cost/loss
        
        Parameters
        ----------
        coef : array
            a np.array used for dot product with feature matrix
        X : dataframe
            a pd.DataFrame of feature
        y : array/series
            an 1-D array/series of dependent variable
            
        Returns
        -------
        int
            an integer representing the cost/loss of the model
        '''
        estimated = np.dot(X, coef)
        error_squared = (estimated.squeeze() - y)**2
        cost = (1/(2*len(y)) * sum(error_squared))
        return cost
    
    @staticmethod
    def gradient_grad_desc(coef, X, y):
        '''
        Method to compute gradient of cost/loss function
        
        Parameters
        ----------
        coef : array
            a np.array used for dot product with feature matrix
        X : dataframe
            a pd.DataFrame of feature
        y : array/series
            an 1-D array/series of dependent variable
        
        Returns
        -------
        array
            an array of shape 1 x N representing rate change for each feature
        '''
        estimated = np.dot(X, coef)
        difference = (estimated.squeeze() - y)
        small_change = X.multiply(difference, axis = 0)
        return 2*small_change.mean()
    
    @staticmethod
    def normalized(feature):
        '''
        Method to normalized feature
        
        Parameters
        ----------
        feature: dataframe
            a pd.Dataframe of independent variable
        
        Returns
        -------
        list
            a list of 3 elements (pd.DataFrame, int, int)
        '''
        mean = feature.mean()
        sd = feature.std()
        norm = (feature - mean)/sd
        return norm, mean, sd
    
if __name__ == "__main__":
    print("Running ml_optimizer file")