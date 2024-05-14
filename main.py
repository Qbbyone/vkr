import utils
import metrics
import pantomkins
import ppg_analyzer
import numpy as np
import matplotlib.pyplot as plt

# init classes
ut = utils.Utils()
pan_tomkins = pantomkins.Pan_tompkins
ppg = ppg_analyzer.Ppg_analyzer

# files
# ut.create_ecg_input_data("pat1_2")

# find qrs
ecg_data = np.loadtxt("./signals/input_data/ecg/ecg1_1.txt", dtype="float")

ecg_sample = ecg_data[6000:26000]
# ecg_sample = ecg

# [qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg_sample, 200, True).run()
# [qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg_sample, 400, True).run()
# print(qrs_amp_raw)
# print(qrs_i_raw)

# calc metrics for ecg
# ref_qrs = np.loadtxt("./signals/ref_data/ref_qrs1_1.txt", dtype="float")
# print(ref_qrs)

# find min-max ppg (photoplethysmogram)
ppg_data = np.loadtxt("./signals/input_data/ppg/ppg1_1.txt", dtype="float")
ppg_sample = ppg_data[6000:26000]
# ppg_sample = ppg_data
[min_max_amp] = ppg(ppg_sample, 200).run()

# calc metrics for ppg


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
