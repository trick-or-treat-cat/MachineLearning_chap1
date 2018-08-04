import numpy as np
import matplotlib.pyplot as plt
from mymodule import model, dataset

np.random.seed(42) # If value of seed is same,, result is always same. 

def J(W, X, y, gamma):
	return np.log(1 + np.exp(-y * W.T.dot(X))).sum() + gamma * W.T.dot(W)

gamma = 5
eta = 1 / (2 * 10 * gamma)

X, y = dataset.dataset2()

BSG = model.BatchSteepestGradientModel(eta=eta, gamma=gamma)
NBM = model.NewtonBasedModel(eta=eta, gamma=gamma)

W_BSG = BSG.fit(X, y)
W_NBM = NBM.fit(X, y)

W_BSG_J = np.array([J(W, X.T, y.reshape(1, -1), gamma).reshape(-1) for W in W_BSG])
W_NBM_J = np.array([J(W, X.T, y.reshape(1, -1), gamma).reshape(-1) for W in W_NBM])

plt.plot(range(100), W_BSG_J[:100], "y", label="batch steepest gradient method")
plt.plot(range(100), W_NBM_J[:100], "c", label="Newton based method")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("$J(W)$")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu")
plt.show()

print(BSG.score(X, y), NBM.score(X, y))
