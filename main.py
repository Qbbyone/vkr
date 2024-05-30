import utils
import metrics
import ecg_analyzer
import ppg_analyzer
import test_ppg
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import pandas as pd
import seaborn as sns


class BP:
    def __init__(self):
        self.ut = utils.Utils()
        self.ecg = ecg_analyzer.Ecg_analyzer
        self.ppg = ppg_analyzer.PPGAnalyzer
        self.ppg_new = test_ppg.PPG

    def create_file(self):
        self.ut.create_ecg_input_data("pat1_2")

    def load_signals_data(self):
        ecg_data = np.loadtxt("./signals/input_data/ecg/ecg1_1.txt", dtype="float")
        self.ecg_sample = ecg_data[7000:34000]

        # self.ut.create_ecg_input_data("pat3_1")
        # self.ut.create_fpg_input_data("pat3_1")

        ppg_data = np.loadtxt("./signals/input_data/ppg/ppg1_1.txt", dtype="float")
        self.ppg_sample = ppg_data
        # self.ppg_sample = ppg_data[7000:34000]

    def plot_both_signals(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.subplot(211)
        plt.plot(self.ecg_sample)
        plt.subplot(212)
        plt.plot(self.ppg_sample)
        plt.show()

    def get_data_from_signals(self):
        # ECG
        [qrs_i, qrs_c, ecg_m, amp] = self.ecg(self.ecg_sample, 1000, True).run()
        # [qrs_amp_raw, qrs_i_raw, delay] = self.ecg(self.ecg_sample, 400, True).run()

        self.ecg_amp = np.array(amp)

        """
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.subplot(211)
        plt.plot(self.ecg_sample)
        plt.plot(ecg_m)
        plt.scatter(qrs_i, qrs_c, color="m")
        plt.subplot(212)
        plt.plot(self.ppg_sample)
        plt.show()
        """

        # PPG
        [s, d, amp] = self.ppg(self.ppg_sample, 1000.0, False).run()

        self.ppg_max = s
        self.ppg_min = d
        self.ppg_amp = np.array(amp)

    def get_signals_processing_metrics(self):
        ref_r = np.loadtxt("./signals/ref_data/ref_qrs1_1.txt", dtype="float")
        # print(ref_r)

    def get_params_h(self):
        r = np.loadtxt("./data/data1_1.txt", dtype="int")
        p = np.loadtxt("./data/data1_2.txt", dtype="int")
        m = np.loadtxt("./data/data1_3.txt", dtype="int")

        int_MInP = np.array([abs(p[i] - m[i]) for i in range(len(p))])
        int_RMin = np.array([abs(r[i] - m[i]) for i in range(len(r))])
        int_RP = np.array([abs(r[i] - p[i]) for i in range(len(r))])
        ampP = np.array(self.ppg_amp)
        ampR = np.array(self.ecg_amp)

        """
        df = pd.DataFrame(
            {
                "int_MInP": int_MInP,
                "int_RMin": int_RMin,
                "int_RP": int_RP,
                "ampP": ampP,
                "ampR": ampR,
            }
        )
        df.to_csv("./params/params_test.csv", index=False)
        """

    def get_params(self):
        int_MInP = []
        int_RMin = []
        int_RP = []
        ampP = []
        ampR = []

        s = [1596, 3314, 5036, 6742, 10129, 11833, 13543, 15249, 16946, 18664]
        d = [1231, 2917, 4626, 6337, 9726, 11437, 13140, 14846, 16556, 18289]
        minP = [s[i] - d[i] for i in range(len(d))]

    def handle_params(self):
        df = pd.read_csv("./params/params_test.csv")
        df.head()
        # расчет основных описательных статистик числовых переменных
        df.describe()

        # визуализация
        num_columns = []
        for column_name in df.columns:
            if df[column_name].dtypes != object:
                num_columns += [column_name]

        width = 2
        height = int(np.ceil(len(num_columns) / width))
        fig, ax = plt.subplots(nrows=height, ncols=width, figsize=(20, 16))
        for idx, column_name in enumerate(num_columns):
            plt.subplot(height, width, idx + 1)
            sns.histplot(data=df, x=column_name, bins=20)

        fig = plt.figure(figsize=(15, 15))
        sns.pairplot(
            data=df,
            palette="bwr",
        )

        # correlation
        cm = sns.color_palette("vlag", as_cmap=True)
        df.corr().style.background_gradient(cmap=cm, vmin=-1, vmax=1)

        # Нормализация и стандартизация
        df_stand = (df - df.mean()) / df.std()

    def run(self):
        self.load_signals_data()
        self.plot_both_signals()
        self.get_data_from_signals()
        # self.get_signals_processing_metrics()
        self.get_params_h()
        self.handle_params()


bp = BP()
bp.run()
