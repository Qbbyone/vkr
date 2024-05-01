import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, convolve, butter, filtfilt


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
        fig, ax = plt.subplots(3, 2)
        if self.fs == 200:
            ax1 = plt.subplot(321)
            ax1.plot(self.ecg)
            ax1.axis("tight")
            ax1.set_title("Raw ECG Signal")

        else:
            plt.subplot(3, 2, (1, 2))
            plt.plot(self.ecg)
            plt.axis("tight")
            plt.title("Raw ECG Signal")

    def filter_signal(self):
        print("filtering")
        if self.fs == 200:
            # Low Pass Filter H(z) = ((1 - z^(-6))^2)/(1 - z^(-1))^2
            b = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
            a = [1, -2, 1]
            h_list = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            h_l = lfilter(b, a, [float(i) for i in h_list])
            ecg_l = convolve(self.ecg, h_l)
            ecg_l = ecg_l / np.max(np.abs(ecg_l))
            self.delay = 6
            if self.gr:
                ax2 = plt.subplot(322)
                ax2.plot(ecg_l)
                ax2.axis("tight")
                ax2.set_title("Low pass filtered")

            # High Pass filter H(z) = (-1+32z^(-16)+z^(-32))/(1+z^(-1))
            b = [
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                32,
                -32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ]
            a = [1, -1]
            h_h_list = [1] + [0] * 32
            h_h = lfilter(b, a, [float(i) for i in h_h_list])
            ecg_h = convolve(self.ecg, h_h)
            ecg_h = ecg_h / np.max(np.abs(ecg_h))
            self.delay += 16  # 16 samples for highpass filtering
            if self.gr:
                ax3 = plt.subplot(323)
                ax3.plot(ecg_h)
                ax3.axis("tight")
                ax3.set_title("High Pass Filtered")
        else:
            # bandpass filter for Noise cancelation of other sampling frequencies(Filtering)
            f1 = 5  # cuttoff low frequency to get rid of baseline wander
            f2 = 15
            Wn = [f1 * 2 / self.fs, f2 * 2 / self.fs]  # cutoff based on fs
            N = 3  # order of 3 less processing
            a, b = butter(N, Wn, btype="band")  # bandpass filtering
            ecg_h = filtfilt(a, b, self.ecg)
            ecg_h = ecg_h / max(abs(ecg_h))
            if self.gr:
                ax3 = plt.subplot(323)
                ax3.plot(ecg_h)
                ax3.axis("tight")
                ax3.set_title("Band Pass Filtered")

        # derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
        h_d = [i * (1 / 8) for i in [-1, -2, 0, 2, 1]]  # 1/8*fs
        ecg_d = np.convolve(ecg_h, h_d)
        ecg_d = ecg_d / np.max(ecg_d)
        self.delay += 2  # delay of derivative filter 2 samples
        if self.gr:
            ax4 = plt.subplot(324)
            ax4.plot(ecg_d)
            ax4.axis("tight")
            ax4.set_title("Filtered with the derivative filter")

        # Squaring nonlinearly enhance the dominant peaks
        ecg_s = [i**2 for i in ecg_d]
        if self.gr:
            ax5 = plt.subplot(325)
            ax5.plot(ecg_s)
            ax5.axis("tight")
            ax5.set_title("Squared")

        print("filtering end")

        # Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
        # ecg_m = conv(ecg_s, ones(1, round(0.150 * fs)) / round(0.150 * fs))
        ecg_m = np.convolve(
            ecg_s, np.ones(round(0.150 * self.fs)) / round(0.150 * self.fs), mode="full"
        )

        self.delay += 15
        if self.gr:
            ax6 = plt.subplot(326)
            ax6.plot(ecg_m)
            ax6.axis("tight")
            ax6.set_title("Averaged with 30 samples length")
            # Averaged with 30 samples length,Black noise,Green Adaptive Threshold,RED Sig Level,Red circles QRS adaptive threshold

    def run(self):
        self.plot_raw_ecg()
        self.filter_signal()
        plt.show()
        return [self.qrs_amp_raw, self.qrs_i_raw, self.delay]
