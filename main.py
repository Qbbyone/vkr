import utils
import metrics
import ecg_analyzer
import ppg_analyzer
import ppg_new
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

# init classes
ut = utils.Utils()
ecg = ecg_analyzer.Ecg_analyzer
ppg = ppg_analyzer.PPGAnalyzer
ppg_new = ppg_new.PPG

# files
# ut.create_ecg_input_data("pat1_2")

# find qrs
ecg_data = np.loadtxt("./signals/input_data/ecg/ecg1_1.txt", dtype="float")

ecg_sample = ecg_data[5000:25000]
# [qrs_amp_raw, qrs_i_raw, delay] = ecg(ecg_sample, 200, True).run()
# [qrs_amp_raw, qrs_i_raw, delay] = ecg(ecg_sample, 400, True).run()
# print(qrs_amp_raw)
# print(qrs_i_raw)

# calc metrics for ecg
# ref_qrs = np.loadtxt("./signals/ref_data/ref_qrs1_1.txt", dtype="float")
# print(ref_qrs)

# find min-max ppg (photoplethysmogram)
ppg_data = np.loadtxt("./signals/input_data/ppg/ppg1_1.txt", dtype="float")
ppg_sample = ppg_data[5000:25000]
# ppg_sample = ppg_data
[min_max_amp] = ppg(ppg_sample, 1000.0, False).run()
# [min_max_amp] = ppg_analyzer(ppg_sample, 1000.0, True).run()


# TEST PPG CLASS
# sampling_rate = 100  # Assuming a sampling rate of 100 Hz
# ppg_new(ppg_sample, sampling_rate).run()

# analyzer = PPGAnalyzer(ppg_signal, sampling_rate)


# calc metrics for ppg
# print("test")


# test plots
def plot_both_signals():
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplot(211)
    plt.plot(ecg_sample)
    plt.subplot(212)
    plt.plot(ppg_sample)
    plt.show()


# plot_both_signals()


# test
def detect_r_peaks(ecg_signal, fs):
    # Step 1: Bandpass filter to remove noise (5-15 Hz for typical ECG)
    nyquist = 0.5 * fs
    low = 5 / nyquist
    high = 15 / nyquist
    b, a = sp.butter(1, [low, high], btype="band")
    filtered_ecg = sp.filtfilt(b, a, ecg_signal)

    # Step 2: Differentiation
    diff_ecg = np.diff(filtered_ecg)

    # Step 3: Squaring
    squared_ecg = diff_ecg**2

    # Step 4: Moving average
    window_size = int(0.150 * fs)
    averaged_ecg = np.convolve(
        squared_ecg, np.ones(window_size) / window_size, mode="same"
    )

    # Step 5: Find peaks
    peaks, _ = sp.find_peaks(
        averaged_ecg, distance=fs / 2.5, height=np.mean(averaged_ecg)
    )

    return peaks, filtered_ecg


# Example usage
fs = 200  # Example sampling frequency
r_peaks, processed_ecg = detect_r_peaks(ecg_sample, fs)

print(f"Detected R-peaks at: {r_peaks}")

# Plotting the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ecg_sample, label="Original ECG")
plt.plot(ppg_sample, label="Original FPG")
plt.plot(processed_ecg, label="Processed ECG")
plt.plot(r_peaks, processed_ecg[r_peaks], "ro", label="R-peaks")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("ECG Signal with Detected R-peaks")
plt.show()
