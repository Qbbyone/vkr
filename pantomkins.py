import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, convolve, butter, filtfilt, find_peaks


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
            ecg_h = convolve(ecg_l, h_h)
            self.ecg_h = ecg_h / np.max(np.abs(ecg_h))
            self.delay += 16  # 16 samples for highpass filtering
            if self.gr:
                ax3 = plt.subplot(323)
                ax3.plot(self.ecg_h)
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
            self.ecg_h = ecg_h / max(abs(ecg_h))
            if self.gr:
                ax3 = plt.subplot(323)
                ax3.plot(self.ecg_h)
                ax3.axis("tight")
                ax3.set_title("Band Pass Filtered")

        # derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
        h_d = [i * (1 / 8) for i in [-1, -2, 0, 2, 1]]  # 1/8*fs
        ecg_d = np.convolve(self.ecg_h, h_d)
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

        # Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
        # ecg_m = conv(ecg_s, ones(1, round(0.150 * fs)) / round(0.150 * fs))
        self.ecg_m = np.convolve(
            ecg_s, np.ones(round(0.150 * self.fs)) / round(0.150 * self.fs), mode="full"
        )
        # print(self.ecg_m)

        self.delay += 15
        if self.gr:
            ax6 = plt.subplot(326)
            ax6.plot(self.ecg_m)
            ax6.axis("tight")
            ax6.set_title("Averaged with 30 samples length")
            # Averaged with 30 samples length,Black noise,Green Adaptive Threshold,RED Sig Level,Red circles QRS adaptive threshold

    def fiducial_mark(self):
        # FIDUCIAL MARK - The waveform is first processed to produce a set of weighted unit
        # samples at the location of the MWI maxima. This is done in order to localize the QRS
        # complex to a single instant of time. The w[k] weighting is the maxima value.
        # Note : a minimum distance of 40 samples is considered between each R wave
        # since in physiological point of view no RR wave can occur in less than 200 msec distance
        self.pks, self.locs = find_peaks(self.ecg_m, distance=np.round(0.2 * self.fs))
        # print(self.pks)
        # print(self.locs)

    def init_training(self):
        # initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE
        self.THR_SIG = np.max(self.ecg_m[0 : 2 * self.fs]) * (
            1 / 3
        )  # 0.25 of the max amplitude
        self.THR_NOISE = np.mean(self.ecg_m[0 : 2 * self.fs]) * (
            1 / 2
        )  # 0.5 of the mean signal is considered to be noise
        self.SIG_LEV = self.THR_SIG
        self.NOISE_LEV = self.THR_NOISE

        # Initialize bandpath filter threshold (2 seconds of the bandpass signal)
        self.THR_SIG1 = np.max(self.ecg_h[0 : 2 * self.fs]) * (
            1 / 3
        )  # 0.25 of the max amplitude
        self.THR_NOISE1 = np.mean(self.ecg_h[0 : 2 * self.fs]) * (1 / 2)
        self.SIG_LEV1 = self.THR_SIG1
        # Signal level in Bandpassed filter
        self.NOISE_LEV1 = self.THR_NOISE1
        # Noise level in Bandpassed filter

    def find_peaks(self):
        fs = round(0.150 * self.fs)

        for i in range(len(self.pks)):
            # locate the corresponding peak in the filtered signal
            if self.locs[i] - fs >= 1 and self.locs[i] <= len(self.ecg_h):
                y_i, x_i = max(
                    (v, idx)
                    for idx, v in enumerate(
                        self.ecg_h[self.locs[i] - fs : self.locs[i] + 1]
                    )
                )
            else:
                if i == 0:
                    y_i, x_i = max(
                        (v, idx) for idx, v in enumerate(self.ecg_h[: self.locs[i] + 1])
                    )
                    self.ser_back = 1
                elif self.locs[i] >= len(self.ecg_h):
                    y_i, x_i = max(
                        (v, idx)
                        for idx, v in enumerate(self.ecg_h[self.locs[i] - fs :])
                    )

            # update the heart_rate (Two heart rate means one the moste recent and the other selected)
            if len(self.qrs_c) >= 9:
                diffRR = np.diff(self.qrs_i[-9:])  # calculate RR interval
                mean_RR = np.mean(
                    diffRR
                )  # calculate the mean of 8 previous R waves interval
                comp = self.qrs_i[-1] - self.qrs_i[-2]  # latest RR
                if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                    # lower down thresholds to detect better in MVI
                    self.THR_SIG *= 0.5
                    # lower down thresholds to detect better in Bandpass filtered
                    self.THR_SIG1 *= 0.5
                else:
                    self.m_selected_RR = mean_RR  # the latest regular beats mean

            # calculate the mean of the last 8 R waves to make sure that QRS is not
            # missing(If no R detected , trigger a search back) 1.66*mean

            if self.m_selected_RR:
                self.test_m = self.m_selected_RR  # if the regular RR available use it
            elif mean_RR and self.m_selected_RR == 0:
                self.test_m = mean_RR
            else:
                self.test_m = 0

            if self.test_m:
                if (self.locs[i] - self.qrs_i[-1]) >= round(
                    1.66 * self.test_m
                ):  # it shows a QRS is missed
                    # search back and locate the max in this interval
                    pks_temp, locs_temp = np.max(
                        self.ecg_m[
                            self.qrs_i[-1]
                            + round(0.200 * self.fs) : self.locs[i]
                            - round(0.200 * self.fs)
                        ]
                    ), np.argmax(
                        self.ecg_m[
                            self.qrs_i[-1]
                            + round(0.200 * self.fs) : self.locs[i]
                            - round(0.200 * self.fs)
                        ]
                    )
                    locs_temp = (
                        self.qrs_i[-1] + round(0.200 * self.fs) + locs_temp
                    )  # location

                if pks_temp > self.THR_NOISE:
                    self.qrs_c.append(pks_temp)
                    self.qrs_i.append(locs_temp)

                # find the location in filtered sig
                if locs_temp <= len(self.ecg_h):
                    y_i_t, x_i_t = np.max(
                        self.ecg_h[locs_temp - round(0.150 * self.fs) : locs_temp]
                    ), np.argmax(
                        self.ecg_h[locs_temp - round(0.150 * self.fs) : locs_temp]
                    )
                else:
                    y_i_t, x_i_t = np.max(
                        self.ecg_h[locs_temp - round(0.150 * self.fs) :]
                    ), np.argmax(self.ecg_h[locs_temp - round(0.150 * self.fs) :])

                # take care of bandpass signal threshold
                if y_i_t > self.THR_NOISE1:
                    self.qrs_i_raw.append(
                        locs_temp - round(0.150 * self.fs) + (x_i_t - 1)
                    )  # save index of bandpass
                    self.qrs_amp_raw.append(y_i_t)  # save amplitude of bandpass
                    self.SIG_LEV1 = (
                        0.25 * y_i_t + 0.75 * self.SIG_LEV1
                    )  # when found with the second threshold
                    self.not_nois = 1
                    self.SIG_LEV = (
                        0.25 * pks_temp + 0.75 * self.SIG_LEV
                    )  # when found with the second threshold
                else:
                    self.not_nois = 0

    def run(self):
        self.plot_raw_ecg()
        self.filter_signal()
        self.fiducial_mark()
        self.init_training()
        # self.find_peaks()
        plt.show()
        return [self.qrs_amp_raw, self.qrs_i_raw, self.delay]
