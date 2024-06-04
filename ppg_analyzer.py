import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


class PPGAnalyzer:
    def __init__(self, ppg_signal, fs, gr=True, gr_th=True):
        self.ppg = ppg_signal
        self.fs = fs
        self.gr = gr
        self.gr_th = gr_th
        self.ppg_filt = None
        self.fidp = None
        self.min_max_amp = []
        self.amp = []

    def plot_raw_ppg(self):
        plt.figure()
        plt.plot(self.ppg, label="Raw PPG Signal")
        plt.legend()
        plt.title("Raw PPG Signal")
        plt.show()

    def prep_ppg(self):
        offset = 3200
        scale = 30
        Araw = self.ppg * scale - offset

        # Filter coefficients
        b1 = [1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 1]
        a1 = [1, -2, 1]

        # Apply the filter
        A = sp.lfilter(b1, a1, Araw) / 24 + 30
        A = (A[3:] + offset) / scale  # Adjust for 0-based indexing in Python

        # High-pass Butterworth filter
        b, a = sp.butter(4, 0.2 / (self.fs / 2), "high")
        p2 = sp.lfilter(b, a, A)

        # Low-pass FIR filter
        Num = sp.firwin(21, 20 / (self.fs / 2), window="hamming")
        A = sp.lfilter(Num, 1, A)

        p2 = p2 + 50

        if self.gr_th:
            plt.figure()
            plt.plot(self.ppg, label="Сигнал до фильтрации")
            plt.plot(p2, label="Сигнал после ФВЧ")
            plt.plot(A, label="Сигнал после ФНЧ")
            plt.grid(which="minor")
            plt.legend(loc="upper left")
            plt.title("Предварительная обработка сигнала ФПГ")
            plt.show()

        return A, p2

    def filter_ppg(self):
        """
        Fs = self.fs  # Sampling Frequency (Hz)
        Fn = Fs / 2  # Nyquist Frequency (Hz)
        Wp = 24.5 / Fn  # Stopband Frequency (Normalized)
        Ws = 25.0 / Fn  # Passband Frequency (Normalized)
        Rp = 1  # Passband Ripple (dB)
        Rs = 50  # Stopband Ripple (dB)

        n, Ws = sp.cheb2ord(Wp, Ws, Rp, Rs)  # Filter Order
        z, p, k = sp.cheby2(
            n, Rs, Ws, btype="high", output="zpk"
        )  # Filter Design, Specify Bandstop
        sos = sp.zpk2sos(z, p, k)  # Convert To Second-Order-Section For Stability
        filtered_ppg = sp.sosfiltfilt(sos, self.ppg)  # Filter Signal

        h_d = [1 / 8, -1 / 8, -2 / 8, 0, 2 / 8, 1 / 8]
        self.derivative_ppg = np.convolve(filtered_ppg, h_d, mode="same")
        """

        nyquist = 0.5 * self.fs
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        b, a = sp.butter(1, [low, high], btype="band")
        self.filtered_ppg = sp.filtfilt(b, a, self.ppg)

        # Smoothing the signal (optional)
        self.smoothed_ppg = np.convolve(
            self.filtered_ppg, np.ones(15) / 15, mode="same"
        )

        if self.gr_th:
            plt.figure()
            plt.plot(self.ppg, label="Сигнал до фильтрации")
            plt.plot(self.filtered_ppg, label="Сигнал после фильтрации")
            plt.legend(loc="upper left")
            plt.title("Фильтрация сигнала ФПГ")
            plt.show()

            plt.figure()
            plt.plot(self.filtered_ppg, label="Сигнал после фильтрации")
            plt.plot(self.smoothed_ppg, label="Сглаженный сигнал")
            plt.legend(loc="upper left")
            plt.title("Сглаживание сигнала ФПГ")
            plt.show()

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

        # amp
        d = np.loadtxt("./data/data1_2.txt", dtype="int")
        # self.amp = self.smoothed_ppg[d]

        # Plot the results

        if self.gr_th:
            plt.figure()
            plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
            plt.plot(self.smoothed_ppg, label="Сигнал ФПГ после фильтрации")
            plt.plot(
                systolic_peaks,
                self.smoothed_ppg[systolic_peaks],
                "o",
                color="purple",
                label="Пики максимума",
            )
            plt.plot(
                diastolic_peaks,
                self.smoothed_ppg[diastolic_peaks],
                "o",
                color="green",
                label="Пики минимума",
            )
            plt.title("Выделение локальных пиков на сигнале ФПГ")
            plt.legend(loc="upper left")
            plt.show()

        self.s = s_peaks
        self.d = d_peaks

        # Find fiducial points
        """
        onsets, notches = self.find_fiducial_points(
            self.smoothed_ppg, systolic_peaks, fs
        )
        """

        if self.gr_th:
            plt.figure(figsize=(12, 6))
            plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
            plt.plot(self.smoothed_ppg, label="Сигнал ФПГ после фильтрации")
            plt.plot(
                s_peaks,
                self.smoothed_ppg[s_peaks],
                "o",
                color="purple",
                label="Систолические пики",
            )
            plt.plot(
                d_peaks,
                self.smoothed_ppg[d_peaks],
                "o",
                color="green",
                label="Диастолические пики",
            )
            # plt.plot(onsets, self.smoothed_ppg[onsets], "go", label="Onsets")
            # plt.plot(notches, self.smoothed_ppg[notches], "go", label="Дикротические зубцы")
            plt.title("Сигнал ФПГ с выделенными пиками")
            # plt.xlabel("Sample Number")
            # plt.ylabel("Amplitude")
            plt.legend(loc="upper left")
            plt.show()

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
        # if self.gr:
        # self.plot_raw_ppg()
        self.prep_ppg()
        self.filter_ppg()
        self.find_peaks()

        return [self.s, self.d, self.amp]
