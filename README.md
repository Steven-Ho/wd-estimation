# Wasserstein Distance Estimation
This project compares different methods of Wasserstein distance estimation. Wasserstein distance (abbr. WD) is a metric of probability distributions. WD catches the underlying discrepancy from a geometric view when probability distributions do not share same support. In those case when $p(x)=0$ while $p(y)>0$, traditional $f$-divergences such as KL/Jensen-Shannon divergence will not provide sufficient information about the two distributions.
WD is a metric related to optimal transport, reflecting the discrepancy corresponding to our intuition, especially when the supports of distributions do not overlap. However, WD has much less applications on Machine Learning due to the complexity of solving the notoriously hard optimal transport problem. Some research give estimation of WD, from two main perspectives: primal and dual.
## Primal Problem
The solution of primal problem requires a match, or a mapping from one distribution to another, that minimizes the total cost of transport. In discrete domain, this problem can be converted to an LP problem. As we know, without any other assumptions, solving LP maybe considerably hard. Sinkhorn algorithm is designed to directly solve it by iterating. Meanwhile, greedy methods will work, too. 
## Dual Problem
The dual form of the optimal transport problem is a derivation by Rockerfellar-Fenchel dualtiy. This formulation is presented in lots of literature, most importantly, it enables us to use neural networks and SGD to obtain corrspoding functionals.
This project implements some of these estimation methods on experimental data.
