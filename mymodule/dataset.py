# -*- coding: utf-8 -*-
import numpy as np


def dataset1():
    n = 100
    x = 3 * (np.random.rand(n, 2) - 0.5)
    radius = x[:, 0] ** 2 + x[:, 1] ** 2
    y = (radius > 0.7 + 0.1 * np.random.randn(n)
         ) == (radius < 2.2 + 0.1 * np.random.randn(n))
    y = np.array(list(map(lambda x: 1 if x else -1, y)))
    return x, y


def dataset2():
    n = 40
    omega = np.random.randn(1)
    noise = 0.8 * np.random.randn(n)
    x = np.random.randn(n, 2)
    y = omega * x[:, 0] + x[:, 1] + noise > 0
    y = np.array(list(map(lambda x: 1 if x else -1, y)))
    return x, y


def dataset3():
    m = 20
    n = 40
    r = 2
    A = np.dot(np.random.rand(m, r), np.random.rand(r, n))
    ninc = 100
    Q = np.random.permutation(m * n)[:ninc]
    A[Q // n, Q % n] = None
    return A
