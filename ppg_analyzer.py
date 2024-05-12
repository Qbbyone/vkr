import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz, sosfiltfilt


class Ppg_analyzer:

    def __init__(self, ppg, fs):
        self.ppg = ppg
        self.fs = fs

        # return
        self.min_max_amp = []

    def plot_raw_ppg(self):
        fig, ax = plt.subplots(3, 2)
        ax1 = plt.subplot(321)
        ax1.plot(self.ppg)
        ax1.axis("tight")
        ax1.set_title("Raw PPG Signal")

    def filter_ppg(self):
        # filter cut-offs, hertz
        lpf_cutoff = 0.7
        hpf_cutoff = 10

        # create filter
        sos_filter = butter(
            10,
            [lpf_cutoff, hpf_cutoff],
            btype="bp",
            analog=False,
            output="sos",
            fs=self.fs,
        )

        w, h = sosfreqz(sos_filter, 2000, fs=self.fs)

        # filter PPG
        ppg_filt = sosfiltfilt(sos_filter, self.ppg)

        # plot filtered ppg
        ax2 = plt.subplot(322)
        ax2.plot(ppg_filt)
        ax2.axis("tight")
        ax2.set_title("Filtered")

    def run(self):
        print(self.ppg)
        self.plot_raw_ppg()
        self.filter_ppg()

        # show results
        plt.show()
        return [self.min_max_amp]
