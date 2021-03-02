import numpy as np 

class Method(object):
    def __init__(self):
        self.max_iter = 1000

    def estimate(self, As, Bs):
        raise NotImplementedError

    def train(self, As, Bs):
        raise NotImplementedError