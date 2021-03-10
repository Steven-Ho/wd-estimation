import numpy as np 
import argparse
import torch
import importlib
import matplotlib.pyplot as plt 
import methods

def step_to_converge(dist):
    T = dist.shape[0]
    L = 50
    B = np.mean(dist[-L:-1])
    epsilon = 0.001
    for i in range(T-L):
        data = dist[i:i+L]
        mean = np.mean(data)
        std = np.std(data)
        if abs(mean - B)/B < epsilon:
            return i+1
    return T

# Evaluating
for x in ["wgan", "bgrl"]:
    method = getattr(methods, x)

    # 0. Training curve
    d = 2
    N = 100
    dv = 10.0
    mean1 = np.zeros((d,))
    mean2 = np.zeros((d,))
    mean2[0] = dv
    cov = np.eye(d)
    A = np.random.multivariate_normal(mean1, cov, N)
    B = np.random.multivariate_normal(mean2, cov, N)
    m = method(d, N)
    m.train(A, B)
    plt.plot(m.vals)
    plt.xlabel("Training steps")
    plt.ylabel("Estimated distance")
    plt.title("Training curve: "+x)
    plt.savefig("fig/"+x+"-curve.png")
    plt.show()
    plt.clf()

    # 1. linearity
    d = 2
    N = 100
    ldv = np.arange(11)
    K = ldv.shape[0]
    dv = np.exp2(ldv)
    dists = np.zeros((K,))
    mean1 = np.zeros((d,))
    mean2 = np.zeros((d,))
    cov = np.eye(d)
    A = np.random.multivariate_normal(mean1, cov, N)
    for i in range(K):
        mean2[0] = dv[i]
        B = np.random.multivariate_normal(mean2, cov, N)
        m = method(d, N)
        m.train(A, B)
        dists[i] = m.estimate(A, B)
    ldists = np.log2(dists)
    plt.plot(ldv, ldists)
    plt.xlabel("Normal center distance (log scale)")
    plt.ylabel("WD estimated value (log scale)")
    plt.title("Linearity test: "+x)
    plt.savefig("fig/"+x+"-linearity.png")
    plt.show()
    plt.clf()

    # 2. convergence rate
    d = np.array([2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    N = 100
    K = d.shape[0]
    dv = 10.0
    steps = np.zeros((K,))
    for i in range(K):
        mean1 = np.zeros((d[i],))
        mean2 = np.zeros((d[i],))
        mean2[0] = dv
        cov = np.eye(d[i])
        A = np.random.multivariate_normal(mean1, cov, N)
        B = np.random.multivariate_normal(mean2, cov, N)
        m = method(d[i], N)
        m.train(A, B)
        steps[i] = step_to_converge(m.vals)
    plt.plot(d, steps)
    plt.xlabel("Dimensions of distribution")
    plt.ylabel("Steps to convergence")
    plt.title("Convergence test: "+x)
    plt.savefig("fig/"+x+"-convergence.png")
    plt.show()
    plt.clf()