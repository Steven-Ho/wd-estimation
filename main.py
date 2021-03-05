import numpy as np 
import argparse
import torch
import importlib

parser = argparse.ArgumentParser(description="Wasserstein distance estimation comparison")
parser.add_argument('--num_samples', type=int, default=100, help="number of samples from two distributions")
parser.add_argument('--dimensions', type=int, default=2, help="dimensions of multivariate Gaussian")

args = parser.parse_args()

# Data to estimate
N = args.num_samples
d = args.dimensions
dv = 20.0 # deviation of two distributions
mean1 = np.zeros((d,))
mean2 = np.zeros((d,))
mean1[0] = dv
cov = np.eye(d)
A = np.random.multivariate_normal(mean1, cov, N)
B = np.random.multivariate_normal(mean2, cov, N)

# WD estimating
methods = ["wgan", "bgtf"]
for m in methods:
    import methods
    method = getattr(methods, m)(d, N)
    method.train(A, B)
    wd = method.estimate(A, B)