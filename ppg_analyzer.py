import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


class PPGAnalyzer:
    def __init__(self, ppg_signal, fs, gr=True):
        self.ppg = ppg_signal
        self.fs = fs
        self.gr = gr
        self.ppg_filt = None
        self.fidp = None
        self.min_max_amp = []

    def filter_ppg(self):
        # filter cut-offs, hertz
        """
        lpf_cutoff = 0.7
        hpf_cutoff = 10

        # create filter
        sos_filter = sp.butter(
            10,
            [lpf_cutoff, hpf_cutoff],
            btype="bp",
            analog=False,
            output="sos",
            fs=self.fs,
        )

        w, h = sp.sosfreqz(sos_filter, 2000, fs=self.fs)

        # filter PPG
        self.filtered_ppg = sp.sosfiltfilt(sos_filter, self.ppg)
        """
        nyquist = 0.5 * self.fs
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        b, a = sp.butter(1, [low, high], btype="band")
        self.filtered_ppg = sp.filtfilt(b, a, self.ppg)

        # Smoothing the signal (optional)
        smoothed_ppg = np.convolve(self.filtered_ppg, np.ones(5) / 5, mode="same")

        """
        if self.gr:
            plt.figure()
            plt.plot(self.ppg, label="Сигнал до фильтрации")
            plt.plot(self.filtered_ppg, label="Сигнал после фильтрации")
            plt.legend()
            plt.title("Фильтрация сигнала ФПГ")
            plt.show()

            plt.figure()
            plt.plot(self.filtered_ppg, label="Сигнал после фильтрации")
            plt.plot(smoothed_ppg, label="Сглаженный сигнал")
            plt.legend()
            plt.title("Сглаживание сигнала ФПГ")
            plt.show()
        """

    def find_peaks(self):
        fs = 100
        systolic_peaks, _ = sp.find_peaks(
            self.filtered_ppg, distance=self.fs / 2
        )  # distance parameter helps to avoid detecting multiple peaks for one heartbeat

        # Find troughs (diastolic points)
        inverted_signal = -self.filtered_ppg
        diastolic_peaks, _ = sp.find_peaks(inverted_signal, distance=self.fs * 0.5)

        s_peaks_test = []
        for i in range(1, len(systolic_peaks)):
            segment = self.filtered_ppg[systolic_peaks[i - 1] : systolic_peaks[i]]
            systolic_index = np.argmax(segment) + systolic_peaks[i - 1]
            s_peaks_test.append(systolic_index)

        s_peaks = []
        for i in range(1, len(s_peaks_test)):
            segment = self.filtered_ppg[s_peaks_test[i - 1] : s_peaks_test[i]]
            systolic_index = np.argmax(segment) + s_peaks_test[i - 1]
            s_peaks.append(systolic_index)

        d_peaks_test = []
        for i in range(1, len(diastolic_peaks)):
            segment = self.filtered_ppg[diastolic_peaks[i - 1] : diastolic_peaks[i]]
            diastolic_index = np.argmin(segment) + diastolic_peaks[i - 1]
            d_peaks_test.append(diastolic_index)

        d_peaks = []
        for i in range(1, len(d_peaks_test)):
            segment = self.filtered_ppg[d_peaks_test[i - 1] : d_peaks_test[i]]
            diastolic_index = np.argmin(segment) + d_peaks_test[i - 1]
            d_peaks.append(diastolic_index)

        d_peaks = d_peaks[:1] + d_peaks[2:]

        # Plot the results

        plt.figure()

        plt.subplot(211)
        plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
        plt.plot(self.filtered_ppg, label="Сигнал ФПГ после фильтрации")
        plt.plot(
            systolic_peaks,
            self.filtered_ppg[systolic_peaks],
            "o",
            color="purple",
            label="Пики максимума",
        )
        plt.title("Нахождение пиков максимума на сигнале ФПГ")
        plt.legend()

        plt.subplot(212)
        plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
        plt.plot(self.filtered_ppg, label="Сигнал ФПГ после фильтрации")
        plt.plot(
            diastolic_peaks,
            self.filtered_ppg[diastolic_peaks],
            "o",
            color="green",
            label="Пики минимума",
        )
        plt.title("Нахождение пиков минимума на сигнале ФПГ")
        plt.legend()
        plt.show()

        # Find fiducial points
        onsets, notches = self.find_fiducial_points(
            self.filtered_ppg, systolic_peaks, fs
        )

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
        plt.plot(self.filtered_ppg, label="Сигнал ФПГ после фильтрации")
        plt.plot(
            s_peaks,
            self.filtered_ppg[s_peaks],
            "o",
            color="purple",
            label="Систолические пики",
        )
        plt.plot(
            d_peaks,
            self.filtered_ppg[d_peaks],
            "o",
            color="green",
            label="Диастолические пики",
        )
        # plt.plot(onsets, self.filtered_ppg[onsets], "go", label="Onsets")
        # plt.plot(notches, self.filtered_ppg[notches], "go", label="Дикротические зубцы")
        plt.title("Сигнал ФПГ с выделенными пиками")
        # plt.xlabel("Sample Number")
        # plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

        print("----- s_peaks")
        print(s_peaks, len(d_peaks))
        print("----- d_peaks")
        print(d_peaks, len(s_peaks))

        s = [1596, 3314, 5036, 6742, 10129, 11833, 13543, 15249, 16946, 18664]
        d = [1231, 2917, 4626, 6337, 9726, 11437, 13140, 14846, 16556, 18289]
        minP = [s[i] - d[i] for i in range(len(d))]
        # print(minP)

    def find_fiducial_points(self, signal, peaks, fs):
        onsets = []
        notches = []

        for peak in peaks:
            # Find the onset (foot of the pulse)
            # Search in the window before the peak
            search_window = signal[max(0, peak - int(fs / 2)) : peak]
            if len(search_window) > 0:
                onset = np.argmin(search_window) + max(0, peak - int(fs / 2))
                onsets.append(onset)

            # Find the dicrotic notch (secondary peak in the falling edge)
            # Search in the window after the peak
            search_window = signal[peak : min(len(signal), peak + int(fs / 2))]
            if len(search_window) > 0:
                notch_candidates = sp.argrelextrema(search_window, np.less)[0]
                if len(notch_candidates) > 0:
                    notch = notch_candidates[0] + peak
                    notches.append(notch)

        return onsets, notches

    def run(self):
        """
        if self.gr:
            plt.figure()
            plt.plot(self.ppg, label="Raw PPG Signal")
            plt.legend()
            plt.title("Raw PPG Signal")
            plt.show()
        """
        self.filter_ppg()
        self.find_peaks()

        return [self.min_max_amp]
