import numpy as np


class Utils:
    def __init__(self):
        pass

    def create_ecg_input_data(self, file_name):
        if file_name:
            path = f"./raw_data/signals_data/{file_name}.txt"
            data = np.loadtxt(path, dtype="str")
            values = np.delete(data, (0), axis=0)
            ecg = np.array(values[:, [1]], float)
            ecg_flatten = ecg.flatten()
            ecg_file_name = f"ecg{file_name[3:]}"
            np.savetxt(f"signals/input_data/ecg/{ecg_file_name}.txt", ecg_flatten)

    def create_fpg_input_data(self, file_name):
        if file_name:
            path = f"./raw_data/signals_data/{file_name}.txt"
            data = np.loadtxt(path, dtype="str")
            values = np.delete(data, (0), axis=0)
            fpg = np.array(values[:, [2]], float)
            fpg_flatten = fpg.flatten()
            fpg_file_name = f"ppg{file_name[3:]}"
            np.savetxt(f"signals/input_data/ppg/{fpg_file_name}.txt", fpg_flatten)

    def create_ref_ecg_data(self, file_name):
        if file_name:
            path = f"./raw_data/types/{file_name}.cvs"
            data = np.loadtxt(path, dtype="str")
            values = np.delete(data, (0), axis=0)
            fpg = np.array(values[:, [2]], float)
            fpg_flatten = fpg.flatten()
            fpg_file_name = f"fpg{file_name[3:]}"
            np.savetxt(f"signals/input_data/fpg/{fpg_file_name}.txt", fpg_flatten)
