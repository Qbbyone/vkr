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

        if self.gr:
            plt.figure()
            plt.plot(self.ppg, label="Сигнал до фильтрации")
            plt.plot(self.filtered_ppg, label="Сигнал после фильтрации")
            plt.legend()
            plt.title("Фильтрация сигнала ФПГ")
            plt.show()

    def find_peaks(self):
        fs = 100
        systolic_peaks, _ = sp.find_peaks(
            self.filtered_ppg, distance=self.fs / 2
        )  # distance parameter helps to avoid detecting multiple peaks for one heartbeat

        # Plot the results
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.ppg, label="Сигнал ФПГ до фильтрации")
        plt.plot(self.filtered_ppg, label="Сигнал ФПГ после фильтрации")
        plt.plot(
            systolic_peaks,
            self.filtered_ppg[systolic_peaks],
            "o",
            color="purple",
            label="Пики",
        )
        plt.title("Нахождение пиков на сигнале ФПГ")
        plt.legend()
        plt.show()

        # Find fiducial points
        onsets, notches = self.find_fiducial_points(
            self.filtered_ppg, systolic_peaks, fs
        )

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.ppg, label="Original PPG Signal")
        plt.plot(self.filtered_ppg, label="Filtered PPG Signal")
        plt.plot(
            systolic_peaks,
            self.filtered_ppg[systolic_peaks],
            "rx",
            label="Systolic Peaks",
        )
        plt.plot(onsets, self.filtered_ppg[onsets], "go", label="Onsets")
        plt.plot(notches, self.filtered_ppg[notches], "mo", label="Dicrotic Notches")
        plt.title("PPG Signal with Detected Fiducial Points")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.legend()
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
