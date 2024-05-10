import utils
import metrics
import pantomkins
import numpy as np

# files
ut = utils.Utils()
# ut.create_ecg_input_data("pat1_2")

# find qrs
ecg = np.loadtxt("./signals/input_data/ecg/ecg1_1.txt", dtype="float")
pan_tomkins = pantomkins.Pan_tompkins

ecg_sample = ecg[2499:5000]

[qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg_sample, 200, True).run()
# [qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg_sample, 400, True).run()
# print(qrs_amp_raw)
# print(qrs_i_raw)

# calc metrics for ecg
ref_qrs = np.loadtxt("./signals/ref_data/ref_qrs1_1.txt", dtype="float")
print(ref_qrs)

# find min-max fpg
# fpg = np.loadtxt("./signals/input_data/fpg/fpg1.txt", dtype="float")
# calc metrics for fpg
