import numpy as np
import matplotlib.pyplot as plt


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
        print("plot raw signal")
        # if self.fs == 200:
        # figure,  ax(1)=subplot(321);plot(ecg);axis tight;title('Raw ECG Signal')
        # else:
        # figure,  ax(1)=subplot(3,2,[1 2]);plot(ecg);axis tight;title('Raw ECG Signal')

    def cancel_noise(self):
        b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
        a = [1, -2, 1]
        print("filtering")

    def run(self):
        self.plot_raw_ecg()
        self.cancel_noise()
        return [self.qrs_amp_raw, self.qrs_i_raw, self.delay]
