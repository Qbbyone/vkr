import pantomkins
import numpy as np

data = np.loadtxt("./signals/pat1_1.txt", dtype="str")

# get ecg and fpg
ecg = np.loadtxt("./signals/ecg1.txt", dtype="float")
fpg = np.loadtxt("./signals/fpg1.txt", dtype="float")


# find qrs
pan_tomkins = pantomkins.Pan_tompkins

[qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg[2500:5000], 200, True).run()
# [qrs_amp_raw, qrs_i_raw, delay] = pan_tomkins(ecg[2500:5000], 400, True).run()
print(qrs_amp_raw)
