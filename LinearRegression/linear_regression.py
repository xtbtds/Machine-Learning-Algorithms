import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge

class MyLinearRegression:
    def __init__(self, fit_intercept=True): 
        self.fit_intercept = fit_intercept   #bias

    def fit(self, X, y):        #calculate the weights
        n, k = X.shape
        X_train = X
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))   #add bias column
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y   #minimized by MLS
        return self
        
    def predict(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
        y_pred = X_train @ self.w
        return y_pred
    
    def get_weights(self):
        return self.w

    
if __name__ == "__main__":

    def linear_expression(x):     #generate test data
        return 5 * x + 6

    objects_num = 50
    X = np.linspace(-5, 5, objects_num)
    y = linear_expression(X) + np.random.randn(objects_num) * 5      #add noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

    plt.figure(figsize=(10, 7))
    plt.plot(X, linear_expression(X), label='real', c='g')
    plt.scatter(X_train, y_train, label='train', c='b')
    plt.scatter(X_test, y_test, label='test', c='orange')

    plt.title("Generated dataset")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

    #train the model
    regressor = MyLinearRegression()
    regressor.fit(X_train[:, np.newaxis], y_train)
    predictions = regressor.predict(X_test[:, np.newaxis])
    w = regressor.get_weights()

    plt.figure(figsize=(20, 7))

    ax = None

    for i, types in enumerate([['train', 'test'], ['train'], ['test']]):
        ax = plt.subplot(1, 3, i + 1, sharey=ax)
        if 'train' in types:
            plt.scatter(X_train, y_train, label='train', c='b')
        if 'test' in types:
            plt.scatter(X_test, y_test, label='test', c='orange')

        plt.plot(X, linear_expression(X), label='real', c='g')
        plt.plot(X, regressor.predict(X[:, np.newaxis]), label='predicted', c='r')

        plt.ylabel('target')
        plt.xlabel('feature')
        plt.title(" ".join(types))
        plt.grid(alpha=0.2)
        plt.legend()

    plt.show()
