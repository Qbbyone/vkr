import numpy as np


class Utils:
    def __init__(self):
        pass

    def create_ecg_file(self, data):
        values = np.delete(data, (0), axis=0)
        ecg = np.array(values[:, [1]], float)
        # fpg = np.array(values[:, [2]], float)
        ecg_flatten = ecg.flatten()

        np.savetxt("ecg1.txt", ecg_flatten)

    def create_fpg_from_raw_data(self, file_name):
        if file_name:
            path = f"./signals/row_data/{file_name}.txt"
            data = np.loadtxt(path, dtype="float")
            values = np.delete(data, (0), axis=0)
            ecg = np.array(values[:, [1]], float)
            ecg_flatten = ecg.flatten()
            ecg_file_name = f'ecg_{file_name}'
            np.savetxt(f"signals/{ecg_file_name}.txt", ecg_flatten)

    def create_fpg_file(self, data):
        values = np.delete(data, (0), axis=0)
