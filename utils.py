import numpy as np

class Utils:
    def __init__(self):
        pass

    def create_ecg_file(self, data):
        values = np.delete(data, (0), axis=0)
        ecg = np.array(values[:, [1]], float)
        #fpg = np.array(values[:, [2]], float)
        ecg_flatten = ecg.flatten()
        np.savetxt("ecg1.txt", ecg_flatten)