import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

class PPG:
    def __init__(self, ppg_signal, sampling_rate):
        self.ppg_signal = ppg_signal
        self.sampling_rate = sampling_rate

    def bandpass_filter(self, lowcut, highcut, order=3):
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, self.ppg_signal)
        return filtered_signal

    def find_fiducial_points(self):
        # Bandpass filter to remove noise and baseline wander
        filtered_ppg = self.bandpass_filter(0.5, 8.0)
        
        # Find peaks (systolic points)
        systolic_peaks, _ = find_peaks(filtered_ppg, distance=self.sampling_rate*0.5)
        
        # Find troughs (diastolic points)
        inverted_signal = -filtered_ppg
        diastolic_troughs, _ = find_peaks(inverted_signal, distance=self.sampling_rate*0.5)
        
        return systolic_peaks, diastolic_troughs

    def plot_signal_with_fiducial_points(self, systolic_peaks, diastolic_troughs):
        plt.figure(figsize=(12, 6))
        plt.plot(self.ppg_signal, label='Original PPG Signal')
        plt.plot(systolic_peaks, self.ppg_signal[systolic_peaks], 'rx', label='Systolic Peaks')
        plt.plot(diastolic_troughs, self.ppg_signal[diastolic_troughs], 'bo', label='Diastolic Troughs')
        plt.xlabel('Sample Index')
        plt.ylabel('PPG Amplitude')
        plt.title('PPG Signal with Fiducial Points')
        plt.legend()
        plt.show()

    def run(self):
        systolic_peaks, diastolic_troughs = self.find_fiducial_points()
        self.plot_signal_with_fiducial_points(systolic_peaks, diastolic_troughs)
    
