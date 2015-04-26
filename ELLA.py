""" Alpha version of a version of ELLA that plays nicely with sklearn

    @author: Paul Ruvolo
"""

import numpy as np
from scipy.linalg import sqrtm, inv, norm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, explained_variance_score

class ELLA(object):
    """ The ELLA model """
    def __init__(self,d,k,base_learner,mu=1,lam=1):
    	""" Initializes a new model for the given base_learner.
    	    d: the number of parameters for the base learner
    	    k: the number of latent model components
    	    base_learner: the base learner to use (currently can only be
    	    		  LinearRegression, Ridge, or LogisticRegression).
    	    mu: the L_1 penalty to use
    	    lam: the L_2 penalty to use
    	    NOTE: currently we don't have a way to specify hyperparameters
            NOTE: currently only binary logistic regression is supported
    	"""
        self.d = d
        self.k = k
        self.L = np.random.randn(d,k)
        self.A = np.zeros((d*k,d*k))
        self.b = np.zeros((d*k,1))
        self.S = np.zeros((k,0))
        self.T = 0
        self.mu = mu
        self.lam = lam
        if base_learner in [LinearRegression, Ridge]:
            self.perf_metric = explained_variance_score
        elif base_learner in [LogisticRegression]:
            self.perf_metric = accuracy_score
        else:
            raise Exception("Unsupported Base Learner")

        self.base_learner = base_learner

    def fit(self,X,y,task_id):
        """ Fit the model to a new batch of training data.  The task_id must
            start at 0 and increase by one each time this function is called.
            Currently you cannot add new data to old tasks.

            X: the training data
            y: the trianing labels
            task_id: the id of the task
        """
        self.T += 1
        single_task_model = self.base_learner(fit_intercept=False).fit(X,y)
        D_t = self.get_hessian(single_task_model, X, y)
        D_t_sqrt = sqrtm(D_t)
        theta_t = single_task_model.coef_

        sparse_encode = Lasso(alpha=self.mu/(X.shape[0]*2.0),
                              fit_intercept=False).fit(D_t_sqrt.dot(self.L),
                                                       D_t_sqrt.dot(theta_t.T))
        self.S = np.hstack((self.S, np.matrix(sparse_encode.coef_).T))
        self.A += np.kron(self.S[:,task_id].dot(self.S[:,task_id].T), D_t)
        self.b += np.kron(self.S[:,task_id].T, np.mat(theta_t).dot(D_t)).T
        L_vectorized = inv(self.A/self.T + self.lam*np.eye(self.d*self.k,self.d*self.k)).dot(self.b)/self.T
        self.L = L_vectorized.reshape((self.k, self.d)).T
        self.revive_dead_components()

    def revive_dead_components(self):
        """ re-initailizes any components that have decayed to 0 """
        for i,val in enumerate(np.sum(self.L,axis=0)):
            if abs(val) < 10**-8:
                self.L[:,i] = np.random.randn(self.d,)

    def predict(self,X,task_id):
        """ Output ELLA's predictions for the specified data on the specified
            task_id.  If using a continuous model (Ridge and LinearRegression)
            the result is the prediction.  If using a classification model
            (LogisticRgerssion) the output is currently a probability.
        """
        if self.base_learner == LinearRegression or self.base_learner == Ridge:
            return X.dot(self.L.dot(self.S[:,task_id]))
        elif self.base_learner == LogisticRegression:
            return 1./(1.0+np.exp(-X.dot(self.L.dot(self.S[:,task_id])))) > 0.5

    def predict_probs(self,X,task_id):
        """ Output ELLA's predictions for the specified data on the specified
            task_id.  If using a continuous model (Ridge and LinearRegression)
            the result is the prediction.  If using a classification model
            (LogisticRgerssion) the output is currently a probability.
        """
        if self.base_learner == LinearRegression or self.base_learner == Ridge:
            raise Exception("This base learner does not support predicting probabilities")
        elif self.base_learner == LogisticRegression:
            return 1./(1.0+np.exp(-X.dot(self.L.dot(self.S[:,task_id]))))


    def score(self,X,y,task_id):
        """ Output the score for ELLA's model on the specified testing data.
            If using a continuous model (Ridge and LinearRegression)
            the score is explained variance.  If using a classification model
            (LogisticRegression) the score is accuracy.
        """
        return self.perf_metric(self.predict(X, task_id),y)

    def get_hessian(self, model, X, y):
        """ ELLA requires that each single task learner provide the Hessian
            of the loss function evaluated around the optimal single task
            parameters.  This funciton implements this for the base learners
            that are currently supported """
        theta_t = model.coef_
        if self.base_learner == LinearRegression:
            return X.T.dot(X)/(2.0*X.shape[0])
        elif self.base_learner == Ridge:
            return X.T.dot(X)/(2.0*X.shape[0]) + model.alpha*np.eye(self.d,self.d)
        elif self.base_learner == LogisticRegression:
            preds = 1./(1.0+np.exp(-X.dot(theta_t.T)))
            per_data_hess = [X[i,:][np.newaxis].T.dot(X[i,:][np.newaxis])*preds[i]*(1-preds[i])/(2.0*X.shape[0]) for i in range(X.shape[0])]
            return np.sum(per_data_hess,axis=0) + np.eye(self.d,self.d)/(2.0*model.C)
