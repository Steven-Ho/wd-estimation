import numpy as np 

class Method(object):
    def __init__(self, max_iter=0):
        self.max_iter = max_iter

    def estimate(self, As, Bs):
        raise NotImplementedError

    def train(self, As, Bs):
        raise NotImplementedError