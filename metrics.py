import numpy as np
from sklearn.metrics import accuracy_score


class Metrics:
    def __init__(self):
        pass

    def accuracy(self, pred, true):
        return accuracy_score(true, pred)
