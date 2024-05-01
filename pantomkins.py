import numpy as np
import matplotlib.pyplot as plt
import scipy


class Pan_tompkins:

    def __init__(self, ecg, fs, gr=True):

        self.ecg = ecg
        self.fs = fs
        self.gr = gr

        # Initialize
        self.qrs_c = []  # amplitude of R
        self.qrs_i = []  # index
        self.SIG_LEV = 0
        self.nois_c = []
        self.nois_i = []

        self.skip = 0  # becomes one when a T wave is detected
        self.not_nois = 0  # it is not noise when not_nois = 1
        self.selected_RR = []  # Selected RR intervals
        self.m_selected_RR = 0
        self.mean_RR = 0

        self.ser_back = 0
        self.test_m = 0
        self.SIGL_buf = []
        self.NOISL_buf = []
        self.THRS_buf = []
        self.SIGL_buf1 = []
        self.NOISL_buf1 = []
        self.THRS_buf1 = []

        # returns ??
        self.qrs_amp_raw = []
        self.qrs_i_raw = []
        self.delay = 0

    def plot_raw_ecg(self):
        # print("plot raw signal")
        if self.fs == 200:
            plt.figure()
            plt.plot(self.ecg)
            plt.title("Raw ECG Signal", fontdict=None, loc="center")
            plt.show()
        else:
            # todo add something to plot
            plt.figure()
            plt.plot(self.ecg)
            plt.title("Raw ECG Signal", fontdict=None, loc="center")
            plt.show()

    def filter_signal(self):
        print("filtering")
        if self.fs == 200:
            # Low Pass Filter H(z) = ((1 - z^(-6))^2)/(1 - z^(-1))^2
            h_l = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0])
            ecg_l = np.convolve(self.ecg, h_l)
            ecg_l = ecg_l / np.max(np.abs(ecg_l))
            delay = 6
            if self.gr:
                plt.figure()
                plt.plot(ecg_l)
                plt.title("Low pass filtered", fontdict=None, loc="center")
                plt.show()

            # High Pass filter H(z) = (-1+32z^(-16)+z^(-32))/(1+z^(-1))
            h_h = np.array(
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    31,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                ]
            )
            ecg_h = np.convolve(self.ecg, h_h)
            ecg_h = ecg_h / np.max(np.abs(ecg_h))
            delay = delay + 16  # 16 samples for highpass filtering
            if self.gr:
                plt.figure()
                plt.plot(ecg_h)
                plt.title("High Pass Filtered", fontdict=None, loc="center")
                plt.show()

        else:
            # bandpass filter
            # for Noise cancelation of other sampling frequencies(Filtering)
            f1 = 5  # cuttoff low frequency to get rid of baseline wander
            f2 = 15
            Wn = (f1 + f2) * 2 / self.fs
        print("filtering end")

    def run(self):
        self.plot_raw_ecg()
        self.filter_signal()
        return [self.qrs_amp_raw, self.qrs_i_raw, self.delay]
