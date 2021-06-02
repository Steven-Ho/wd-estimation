import numpy as np 
import argparse
import torch
import time
import importlib
import methods

parser = argparse.ArgumentParser(description="Wasserstein distance estimation comparison")
parser.add_argument('--num_samples', type=int, default=100, help="number of samples from two distributions")
parser.add_argument('--dimensions', type=int, default=4, help="dimensions of multivariate Gaussian")
parser.add_argument('--method', type=str, default="bgrl", help="method of calculating WD")

args = parser.parse_args()
ms = ["wgan", "bgrl", "swd", "pwd"]
dvs = [2, 16, 64]
res = np.zeros((4, 3, 6, 2))
for i in range(4):
    m = ms[i]
    for j in range(3):
        dv = dvs[j]
        # Data to estimate
        N = args.num_samples
        d = args.dimensions
        # dv = 20 # deviation of two distributions
        mean1 = np.zeros((d,))
        mean2 = np.zeros((d,))
        mean1[0] = dv
        cov = np.eye(d)
        A = np.random.multivariate_normal(mean1, cov, N)
        B = np.random.multivariate_normal(mean2, cov, N)

        # WD estimating
        for k in range(6):
            start_time = time.time()
            method = getattr(methods, m)(d, N)
            method.train(A, B)
            wd = method.estimate(A, B)
            res[i,j,k,0] = wd
            res[i,j,k,1] = time.time() - start_time
            # print(wd)
            # print("{} seconds".format(str(time.time() - start_time)))
means = np.mean(res[:,:,1:,:], axis=2)
stds = np.std(res[:,:,1:,:], axis=2)
np.save("results", res)