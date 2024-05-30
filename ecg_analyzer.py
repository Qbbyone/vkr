import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, convolve, butter, filtfilt, find_peaks, detrend
import scipy.io


class Ecg_analyzer:

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
        self.qrs_locs = []
        self.amp = []

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

    def prep_signal(self):
        fbs = scipy.io.loadmat("./filters/FBS1.mat")
        # Display the numerator and denominator coefficients
        print(fbs["Num"])
        print(fbs["Den"])

        # Apply the filter to the ECG signal
        filt_ecg = lfilter(fbs["Num"].flatten(), fbs["Den"].flatten(), self.ecg)

        filt_ecg = detrend(filt_ecg)

        # Load the filter coefficients from the .mat file
        fhp = scipy.io.loadmat("./filters/FHP2.mat")

        # Apply the filter
        filt_ecg1 = lfilter(fhp["Num"].flatten(), fhp["Den"].flatten(), filt_ecg)

        # Initialize filt_ecg2 with zeros
        filt_ecg2 = np.zeros_like(filt_ecg1)

        # Adjust filt_ecg1 for the moving average calculation
        filt_ecg1 = np.concatenate((np.full(15, filt_ecg1[0]), filt_ecg1))

        # Calculate the moving average
        for i in range(15, len(filt_ecg1)):
            filt_ecg2[i - 15] = np.sum(filt_ecg1[i - 15 : i + 1]) / 16

        # Plotting if gr is True
        """
        if self.gr:
            plt.figure()
            plt.plot(self.ecg, label="ECG")
            plt.plot(filt_ecg, label="Filtered ECG")
            plt.plot(filt_ecg1[15:], label="Filtered ECG1")
            plt.plot(filt_ecg2, label="Filtered ECG2")
            plt.grid(which="minor")
            plt.legend()
            plt.show()
        """

        # Adjust filt_ecg2
        self.prep_ecg = filt_ecg2[9:]

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
        ecg_s = np.square(ecg_d)
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
        idx, _ = find_peaks(self.ecg_m, distance=np.round(0.2 * self.fs))

        pks = [self.ecg_m[i] for i in idx]
        locs = idx
        # print(pks)
        # print(locs)

        self.pks = pks  # Local maxima, returned as a vector of signal values
        self.locs = locs  # a vector of indices (peaks locations)

    def init_training(self):
        # initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE
        """
        self.THR_SIG = np.max(self.ecg_m[0 : 2 * self.fs]) * (
            1 / 3
        )  # 0.25 of the max amplitude
        """

        self.THR_SIG = np.max(self.ecg_m[0 : 2 * self.fs]) * 0.8

        self.THR_NOISE = np.mean(self.ecg_m[0 : 2 * self.fs]) * 0.8
        # 0.5 of the mean signal is considered to be noise

        self.SIG_LEV = self.THR_SIG
        self.NOISE_LEV = self.THR_NOISE

        # Initialize bandpath filter threshold (2 seconds of the bandpass signal)
        """
        self.THR_SIG1 = np.max(self.ecg_h[0 : 2 * self.fs]) * (
            1 / 3
        )  # 0.25 of the max amplitude
        """

        self.THR_SIG1 = np.max(self.ecg_m[0 : 2 * self.fs]) * 0.8

        self.THR_NOISE1 = np.mean(self.ecg_h[0 : 2 * self.fs]) * 0.8

        self.SIG_LEV1 = self.THR_SIG1  # Signal level in Bandpassed filter
        self.NOISE_LEV1 = self.THR_NOISE1  # Noise level in Bandpassed filter

    def find_peaks(self):
        fs_r_150 = round(0.150 * self.fs)
        fs_r_200 = round(0.200 * self.fs)

        for i in range(len(self.pks)):

            # locate the corresponding peak in the filtered signal
            if self.locs[i] - fs_r_150 >= 1 and self.locs[i] <= len(self.ecg_h):
                y_i, x_i = max(
                    (v, idx)
                    for idx, v in enumerate(
                        self.ecg_h[self.locs[i] - fs_r_150 : self.locs[i] + 1]
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
                        for idx, v in enumerate(self.ecg_h[self.locs[i] - fs_r_150 :])
                    )

            # update the heart_rate (Two heart rate means one the most recent and the other selected)
            if len(self.qrs_c) >= 9:
                diffRR = np.diff(self.qrs_i[-9:])  # calculate RR interval
                self.mean_RR = np.mean(
                    diffRR
                )  # calculate the mean of 8 previous R waves interval
                comp = self.qrs_i[-1] - self.qrs_i[-2]  # latest RR
                if comp <= 0.92 * self.mean_RR or comp >= 1.16 * self.mean_RR:
                    # lower down thresholds to detect better in MVI
                    self.THR_SIG *= 0.5
                    # lower down thresholds to detect better in Bandpass filtered
                    self.THR_SIG1 *= 0.5
                else:
                    self.m_selected_RR = self.mean_RR  # the latest regular beats mean

            # calculate the mean of the last 8 R waves to make sure that QRS is not
            # missing(If no R detected , trigger a search back) 1.66*mean
            if self.m_selected_RR:
                self.test_m = self.m_selected_RR  # if the regular RR available use it
            elif self.mean_RR and self.m_selected_RR == 0:
                self.test_m = self.mean_RR
            else:
                self.test_m = 0

            if self.test_m:
                # todo
                """
                if (self.locs[i] - self.qrs_i[-1]) >= round(
                    1.66 * self.test_m
                ):  # it shows a QRS is missed
                    # search back and locate the max in this interval
                    pks_temp, locs_temp = np.max(
                        self.ecg_m[self.qrs_i[-1] + fs_r_200 : self.locs[i] - fs_r_200]
                    ), np.argmax(
                        self.ecg_m[self.qrs_i[-1] + fs_r_200 : self.locs[i] - fs_r_200]
                    )
                    locs_temp = self.qrs_i[-1] + fs_r_200 + locs_temp  # location

                    if pks_temp > self.THR_NOISE:
                        self.qrs_c.append(pks_temp)
                        self.qrs_i.append(locs_temp)

                    # find the location in filtered sig
                    if locs_temp <= len(self.ecg_h):
                        y_i_t, x_i_t = np.max(
                            self.ecg_h[locs_temp - fs_r_150 : locs_temp]
                        ), np.argmax(self.ecg_h[locs_temp - fs_r_150 : locs_temp])
                    else:
                        y_i_t, x_i_t = np.max(
                            self.ecg_h[locs_temp - fs_r_150 :]
                        ), np.argmax(self.ecg_h[locs_temp - fs_r_150 :])

                    # take care of bandpass signal threshold
                    if y_i_t > self.THR_NOISE1:
                        self.qrs_i_raw.append(
                            locs_temp - fs_r_150 + (x_i_t - 1)
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
                """

            # find noise and QRS peaks
            if self.pks[i] >= self.THR_SIG:
                # if a QRS candidate occurs within 360ms of the previous QRS,
                # the algorithm determines if its T wave or QRS
                if len(self.qrs_c) >= 3:
                    if (self.locs[i] - self.qrs_i[-1]) <= round(0.3600 * self.fs):
                        Slope1 = np.mean(
                            np.diff(
                                self.ecg_m[
                                    self.locs[i] - round(0.075 * self.fs) : self.locs[i]
                                ]
                            )
                        )  # mean slope of the waveform at that position
                        Slope2 = np.mean(
                            np.diff(
                                self.ecg_m[
                                    self.qrs_i[-1]
                                    - round(0.075 * self.fs) : self.qrs_i[-1]
                                ]
                            )
                        )  # mean slope of previous R wave
                        if abs(Slope1) <= abs(
                            0.5 * Slope2
                        ):  # slope less then 0.5 of previous R
                            self.nois_c.append(self.pks[i])
                            self.nois_i.append(self.locs[i])
                            self.skip = 1  # T wave identification
                            # adjust noise level in both filtered and MVI
                            self.NOISE_LEV1 = 0.125 * y_i + 0.875 * self.NOISE_LEV1
                            self.NOISE_LEV = (
                                0.125 * self.pks[i] + 0.875 * self.NOISE_LEV
                            )
                    else:
                        self.skip = 0

                if self.skip == 0:  # skip is 1 when a T wave is detected
                    self.qrs_c.append(self.pks[i])
                    self.qrs_i.append(self.locs[i])

                # bandpass filter check threshold
                if y_i >= self.THR_SIG1:
                    if self.ser_back:
                        self.qrs_i_raw.append(x_i)  # save index of bandpass
                    else:
                        self.qrs_i_raw.append(
                            self.locs[i] - round(0.150 * self.fs) + (x_i - 1)
                        )  # save index of bandpass
                    self.qrs_amp_raw.append(y_i)  # save amplitude of bandpass
                    self.SIG_LEV1 = (
                        0.125 * y_i + 0.875 * self.SIG_LEV1
                    )  # adjust threshold for bandpass filtered sig

                # adjust Signal level
                self.SIG_LEV = 0.125 * self.pks[i] + 0.875 * self.SIG_LEV

            elif self.THR_NOISE <= self.pks[i] and self.pks[i] < self.THR_SIG:
                # adjust Noise level in filtered sig
                self.NOISE_LEV1 = 0.125 * y_i + 0.875 * self.NOISE_LEV1
                # adjust Noise level in MVI
                self.NOISE_LEV = 0.125 * self.pks[i] + 0.875 * self.NOISE_LEV

            elif self.pks[i] < self.THR_NOISE:
                self.nois_c.append(self.pks[i])
                self.nois_i.append(self.locs[i])

                # noise level in filtered signal
                self.NOISE_LEV1 = 0.125 * y_i + 0.875 * self.NOISE_LEV1

                # adjust Noise level in MVI
                self.NOISE_LEV = 0.125 * self.pks[i] + 0.875 * self.NOISE_LEV

            # adjust the threshold with SNR
            if self.NOISE_LEV != 0 or self.SIG_LEV != 0:
                self.THR_SIG = self.NOISE_LEV + 0.25 * abs(
                    self.SIG_LEV - self.NOISE_LEV
                )
                self.THR_NOISE = 0.5 * self.THR_SIG

            # adjust the threshold with SNR for bandpassed signal
            if self.NOISE_LEV1 != 0 or self.SIG_LEV1 != 0:
                self.THR_SIG1 = self.NOISE_LEV1 + 0.25 * abs(
                    self.SIG_LEV1 - self.NOISE_LEV1
                )
                self.THR_NOISE1 = 0.5 * self.THR_SIG1

            # take a track of thresholds of smoothed signal
            self.SIGL_buf.append(self.SIG_LEV)
            self.NOISL_buf.append(self.NOISE_LEV)
            self.THRS_buf.append(self.THR_SIG)

            # take a track of thresholds of filtered signal
            self.SIGL_buf1.append(self.SIG_LEV1)
            self.NOISL_buf1.append(self.NOISE_LEV1)
            self.THRS_buf1.append(self.THR_SIG1)

            # reset parameters
            self.skip = 0
            self.not_nois = 0
            self.ser_back = 0

    def plot_peaks(self):
        plt.scatter(self.qrs_i, self.qrs_c, color="m")
        plt.plot(self.locs, self.NOISL_buf, "--k", linewidth=2)
        plt.plot(self.locs, self.SIGL_buf, "--r", linewidth=2)
        plt.plot(self.locs, self.THRS_buf, "--g", linewidth=2)
        plt.show()

    def plot_results(self):
        # Create figure and subplots
        fig, az = plt.subplots(3, 1, sharex=True)

        # First subplot
        az[0].plot(self.ecg_h)
        az[0].scatter(self.qrs_i_raw, self.qrs_amp_raw, color="m")
        az[0].plot(self.locs, self.NOISL_buf1, linewidth=2, linestyle="--", color="k")
        az[0].plot(self.locs, self.SIGL_buf1, linewidth=2, linestyle="-.", color="r")
        az[0].plot(self.locs, self.THRS_buf1, linewidth=2, linestyle="-.", color="g")
        az[0].set_title("QRS on Filtered Signal")
        az[0].axis("tight")

        # Second subplot
        az[1].plot(self.ecg_m)
        az[1].scatter(self.qrs_i, self.qrs_c, color="m")
        az[1].plot(self.locs, self.NOISL_buf, linewidth=2, linestyle="--", color="k")
        az[1].plot(self.locs, self.SIGL_buf, linewidth=2, linestyle="-.", color="r")
        az[1].plot(self.locs, self.THRS_buf, linewidth=2, linestyle="-.", color="g")
        az[1].set_title(
            "QRS on MVI signal and Noise level(black),Signal Level (red) and Adaptive Threshold(green)"
        )
        az[1].axis("tight")

        # Third subplot
        ecg_centered = self.ecg - np.mean(self.ecg)
        az[2].plot(ecg_centered)
        az[2].set_title("Pulse train of the found QRS on ECG signal")
        for qrs in self.qrs_i_raw:
            az[2].plot(
                [qrs, qrs],
                [np.min(ecg_centered) / 2, np.max(ecg_centered) / 2],
                linestyle="-.",
                color="r",
                linewidth=2.5,
            )
        az[2].axis("tight")

        # Link x axes and enable zoom
        plt.subplots_adjust(hspace=0.5)
        plt.get_current_fig_manager().toolbar.zoom()
        plt.show()

    def run(self):
        if self.gr:
            self.plot_raw_ecg()
        # preprocessing
        self.prep_signal()
        # filters
        self.filter_signal()
        self.fiducial_mark()
        # init variables
        self.init_training()
        # find qrs
        self.find_peaks()
        # show plots
        if self.gr:
            self.plot_peaks()
            self.plot_results()

        print(self.qrs_i_raw)

        """
        plt.figure()
        plt.subplot(211)
        plt.plot(ecg_s)
        plt.title("Квадрат сигнала")
        plt.subplot(212)
        plt.plot(self.ecg_m)
        plt.title("Усреднение по скользящему окну")
        plt.show()
        """

        plt.figure()
        plt.plot(self.ecg)
        plt.plot(self.ecg_m, label="Усредненный сигнал")
        plt.scatter(self.qrs_i, self.qrs_c, color="m")
        plt.plot(
            self.locs,
            self.NOISL_buf,
            linewidth=2,
            linestyle="--",
            color="k",
            label="Уровень шума",
        )
        plt.plot(
            self.locs,
            self.SIGL_buf,
            linewidth=2,
            linestyle="-.",
            color="r",
            label="Уровень сигнала",
        )
        plt.plot(
            self.locs,
            self.THRS_buf,
            linewidth=2,
            linestyle="-.",
            color="g",
            label="Адаптивный порог",
        )
        plt.legend()
        plt.title("QRS-комплексы на усредненном сигнале")
        plt.show()

        d = np.loadtxt("./data/data1_1.txt", dtype="int")
        self.amp = self.ecg_m[d]

        return [self.qrs_i, self.qrs_c, self.ecg_m, self.amp]

    def test_plot(self, a, b, label_a, label_b, title):
        # plot
        plt.figure()
        plt.plot(a, label=label_a)
        plt.plot(b, label=label_b)
        plt.title(title)
        plt.legend()
        plt.show()
