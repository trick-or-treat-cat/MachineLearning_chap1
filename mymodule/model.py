# -*- coding: utf-8 -*-
import numpy as np


class BatchSteepestGradientModel(object):
    def __init__(self, eta=0.5, gamma=1.0):
        self.W = 0
        self.eta = eta
        self.gamma = gamma

    def fit(self, X, Y, epochs=100):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        else:
            X = X.T

        if len(Y.shape) == 1:
            Y = Y.reshape(1, -1)
        else:
            Y = Y.T

        self.W = np.random.randn(
            X.shape[0], Y.shape[0])

        Ws = np.array([self.W])

        for epoch in range(epochs):
            self.W -= (self.eta
                       * (((-Y * X * np.exp(-Y * np.dot(self.W.T, X)))
                            / (1 + np.exp(-Y * np.dot(self.W.T, X)))).sum()
                       + 2 * self.gamma * self.W))
            Ws = np.append(Ws, [self.W], axis=0)

        return Ws

    def predict(self, X):
        if len(X.shape) == 1:
            X.reshape(-1, 1)
        else:
            X = X.T

        Y_pred = np.dot(
            self.W.T, X)
        Y_pred = Y_pred > 0
        Y_pred = np.array(
            list(map(lambda x: 1 if x else -1, Y_pred.reshape(-1))))

        return Y_pred

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return (Y == Y_pred).sum() / len(Y)


class NewtonBasedModel(object):
    def __init__(self, eta=0.5, gamma=1):
        self.W = 0
        self.eta = eta
        self.gamma = gamma

    def fit(self, X, Y, epochs=100):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        else:
            X = X.T

        if len(Y.shape) == 1:
            Y = Y.reshape(1, -1)
        else:
            Y = Y.T

        self.W = np.random.randn(
            X.shape[0], Y.shape[0])

        Ws = np.array([self.W])

        for epoch in range(epochs):
            hessian = (((Y ** 2 * np.array([np.outer(xi, xi) for xi in X.T]).T 
                         * np.exp(-Y * self.W.T.dot(X)))
                         / (1 + np.exp(-Y * self.W.T.dot(X))) ** 2).sum()
                         + 2 * self.gamma * np.eye(X.shape[0]))
            grad = (((-Y * X * np.exp(-Y * self.W.T.dot(X)))
					  / (1 + np.exp(-Y * self.W.T.dot(X)))).sum()
					  + 2 * self.gamma * self.W)
            self.W -= self.eta * np.linalg.inv(hessian).dot(grad)
            Ws = np.append(Ws, [self.W], axis=0)

        return Ws

    def predict(self, X):
        if len(X.shape) == 1:
            X.reshape(-1, 1)
        else:
            X = X.T

        Y_pred = np.dot(
            self.W.T, X)
        Y_pred = Y_pred > 0
        Y_pred = np.array(
            list(map(lambda x: 1 if x else -1, Y_pred.reshape(-1))))

        return Y_pred

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return (Y == Y_pred).sum() / len(Y)


if __name__ == "__main__":
    gamma = int(input("hyper-parameter: "))
    clf = NewtonBasedModel(eta=1 / (2 * gamma), gamma=gamma)
    W = clf.fit(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            np.array([-1, 1, 1, 1]))
    print(clf.score(np.array([[0, 0], [0, 1], [1, 0], [
          1, 1], [-1, -1], [0.1, 0.1]]), np.array([-1, 1, 1, 1, -1, 1])))
    print(W)
