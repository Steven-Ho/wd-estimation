import numpy as np 
import argparse
import torch
import importlib
import methods

parser = argparse.ArgumentParser(description="Wasserstein distance estimation comparison")
parser.add_argument('--num_samples', type=int, default=100, help="number of samples from two distributions")
parser.add_argument('--dimensions', type=int, default=5, help="dimensions of multivariate Gaussian")
parser.add_argument('--method', type=str, default="swd", help="method of calculating WD")

args = parser.parse_args()

# Data to estimate
N = args.num_samples
d = args.dimensions
dv = 10.0 # deviation of two distributions
mean1 = np.zeros((d,))
mean2 = np.zeros((d,))
mean1[0] = dv
cov = np.eye(d)
A = np.random.multivariate_normal(mean1, cov, N)
B = np.random.multivariate_normal(mean2, cov, N)

# WD estimating
method = getattr(methods, args.method)(d, N)
method.train(A, B)
wd = method.estimate(A, B)
print(wd)