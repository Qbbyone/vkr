import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp


class Ppg_analyzer:

    def __init__(self, ppg, fs, gr=True):
        self.ppg = ppg
        self.fs = fs
        self.gr = gr

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
        self.ppg_filt = sp.sosfiltfilt(sos_filter, self.ppg)

        # plot filtered ppg
        if self.gr:
            ax2 = plt.subplot(322)
            ax2.plot(self.ppg_filt)
            ax2.axis("tight")
            ax2.set_title("Filtered")

    def pulse_detect(self, x, fs, w):
        # d2max algoritm by default

        # Pre-processing of signal
        x_d = sp.detrend(x)
        sos = sp.butter(10, [0.5, 10], btype="bp", analog=False, output="sos", fs=fs)
        x_f = sp.sosfiltfilt(sos, x_d)

        # Peak detection in windows of length w
        n_int = np.floor(len(x_f) / (w * fs))
        for i in range(int(n_int)):
            start = i * fs * w
            stop = (i + 1) * fs * w - 1
            # print('Start: ' + str(start) + ', stop: ' + str(stop) + ', fs: ' + str(fs))
            aux = x_f[range(start, stop)]
            locs = self.d2max(aux, fs)
            locs = locs + start
            if i == 0:
                ibis = locs
            else:
                ibis = np.append(ibis, locs)

        if n_int * fs * w != len(x_f):
            start = stop + 1
            stop = len(x_f)
            aux = x_f[range(start, stop)]

            if len(aux) > 20:
                locs = self.d2max(aux, fs)
                locs = locs + start
                ibis = np.append(ibis, locs)

        (ind,) = np.where(ibis <= len(x_f))
        ibis = ibis[ind]

        ibis = self.peak_correction(x, ibis, fs, 20, 5, [0.5, 1.5])

        # fig = plt.figure()
        # plt.plot(x)
        # plt.plot(x_d)
        # plt.plot(x_f)
        # plt.scatter(ibis,x_f[ibis],marker = 'o',color = 'red')
        # plt.scatter(ibis,x[ibis],marker = 'o',color = 'red')

        return ibis
    
    def peak_correction(self,x,locs,fs,t,stride,th_len):
        #fig = plt.figure()
        #plt.plot(x)
        #plt.scatter(locs,x[locs],marker = 'o',color = 'red', label = 'Original')
        #plt.title('Peak correction')

        # Correction of long and short IBIs
        len_window = np.round(t*fs)
        #print('Window length: ' + str(len_window))
        first_i = 0
        second_i = len_window - 1

        while second_i < len(x):
            ind1, = np.where(locs >= first_i)
            ind2, = np.where(locs <= second_i)
            ind = np.intersect1d(ind1, ind2)

            win = locs[ind]
            dif = np.diff(win)
            #print('Indices: ' + str(ind) + ', locs: ' + str(locs[ind]) + ', dif: ' + str(dif))

            th_dif = np.zeros(2)
            th_dif[0] = th_len[0]*np.median(dif)
            th_dif[1] = th_len[1]*np.median(dif)

            th_amp = np.zeros(2)
            th_amp[0] = 0.75*np.median(x[win])
            th_amp[1] = 1.25*np.median(x[win])
            #print('Length thresholds: ' + str(th_dif) + ', amplitude thresholds: ' + str(th_amp))

            j = 0

            while j < len(dif):
                if dif[j] <= th_dif[0]:
                    if j == 0:
                        opt = np.append(win[j], win[j + 1])
                    else:
                        opt = np.append(win[j], win[j + 1]) - win[j - 1]
                    print('Optional: ' + str(opt))
                    dif_abs = np.abs(opt - np.median(dif))
                    min_val = np.min(dif_abs)
                    ind_min, = np.where(dif_abs == min_val)
                    print('Minimum: ' + str(min_val) + ', index: ' + str(ind_min))
                    if ind_min == 0:
                        print('Original window: ' + str(win), end = '')
                        win = np.delete(win, win[j + 1])
                        print(', modified window: ' + str(win))
                    else:
                        print('Original window: ' + str(win), end = '')
                        win = np.delete(win, win[j])
                        print(', modified window: ' + str(win))
                    dif = np.diff(win)
                elif dif[j] >= th_dif[1]:
                    aux_x = x[win[j]:win[j + 1]]
                    locs_pks, _ = sp.find_peaks(aux_x)
                    #fig = plt.figure()
                    #plt.plot(aux_x)
                    #plt.scatter(locs_pks,aux_x[locs_pks],marker = 'o',color = 'red')

                    locs_pks = locs_pks + win[j]
                    ind1, = np.where(x[locs_pks] >= th_amp[0])
                    ind2, = np.where(x[locs_pks] <= th_amp[1])
                    ind = np.intersect1d(ind1, ind2)
                    locs_pks = locs_pks[ind]
                    #print('Locations: ' + str(locs_pks))

                    if len(locs_pks) != 0:
                        opt = locs_pks - win[j]

                        dif_abs = np.abs(opt - np.median(dif))
                        min_val = np.min(dif_abs)
                        ind_min, = np.where(dif_abs == min_val)

                        win = np.append(win, locs_pks[ind_min])
                        win = np.sort(win)
                        dif = np.diff(win)
                        j = j + 1
                    else:
                        opt = np.round(win[j] + np.median(dif))
                        if opt < win[j + 1]:
                            win = np.append(win, locs_pks[ind_min])
                            win = np.sort(win)
                            dif = np.diff(win)
                            j = j + 1
                        else:
                            j = j + 1
                else:
                    j = j + 1

            locs = np.append(win, locs)
            locs = np.sort(locs)

            first_i = first_i + stride*fs - 1
            second_i = second_i + stride*fs - 1

        dif = np.diff(locs)
        dif = np.append(0, dif)
        ind, = np.where(dif != 0)
        locs = locs[ind]

        #plt.scatter(locs,x[locs],marker = 'o',color = 'green', label = 'After length correction')

        # Correction of points that are not peaks
        i = 0
        pre_loc = 0
        while i < len(locs):
            if locs[i] == 0:
                locs = np.delete(locs, locs[i])
            elif locs[i] == len(x):
                locs = np.delete(locs, locs[i])
            else:
                #print('Previous: ' + str(x[locs[i] - 1]) + ', actual: ' + str(x[locs[i]]) + ', next: ' + str(x[locs[i] + 1]))
                cond = (x[locs[i]] >= x[locs[i] - 1]) and (x[locs[i]] >= x[locs[i] + 1])
                #print('Condition: ' + str(cond))
                if cond:
                    i = i + 1
                else:
                    if locs[i] == pre_loc:
                        i = i + 1
                    else:
                        if i == 0:
                            aux = x[0:locs[i + 1] - 1]
                            aux_loc = locs[i] - 1
                            aux_start = 0
                        elif i == len(locs) - 1:
                            aux = x[locs[i - 1]:len(x) - 1]
                            aux_loc = locs[i] - locs[i - 1]
                            aux_start = locs[i - 1]
                        else:
                            aux = x[locs[i - 1]:locs[i + 1]]
                            aux_loc = locs[i] - locs[i - 1]
                            aux_start = locs[i - 1]
                    
                        #print('i ' + str(i) + ' out of ' + str(len(locs)) + ', aux length: ' + str(len(aux)) +
                        #      ', location: ' + str(aux_loc))
                        #print('Locs i - 1: ' + str(locs[i - 1]) + ', locs i: ' + str(locs[i]) + ', locs i + 1: ' + str(locs[i + 1]))

                        pre = self.find_closest_peak(aux, aux_loc, 'backward')
                        pos = self.find_closest_peak(aux, aux_loc, 'forward')
                        #print('Previous: ' + str(pre) + ', next: ' + str(pos) + ', actual: ' + str(aux_loc))

                        ibi_pre = np.append(pre - 1, len(aux) - pre)
                        ibi_pos = np.append(pos - 1, len(aux) - pos)
                        ibi_act = np.append(aux_loc - 1, len(aux) - aux_loc)
                        #print('Previous IBIs: ' + str(ibi_pre) + ', next IBIs: ' + str(ibi_pos) +
                        #      ', actual IBIs: ' + str(ibi_act))

                        dif_pre = np.abs(ibi_pre - np.mean(np.diff(locs)))
                        dif_pos = np.abs(ibi_pos - np.mean(np.diff(locs)))
                        dif_act = np.abs(ibi_act - np.mean(np.diff(locs)))
                        #print('Previous DIF: ' + str(dif_pre) + ', next DIF: ' + str(dif_pos) +
                        #      ', actual DIF: ' + str(dif_act))

                        avgs = [np.mean(dif_pre), np.mean(dif_pos), np.mean(dif_act)]
                        min_avg = np.min(avgs)
                        ind, = np.where(min_avg == avgs)
                        #print('Averages: ' + str(avgs) + ', min index: ' + str(ind))
                        if len(ind) != 0:
                            ind = ind[0]

                        if ind == 0:
                            locs[i] = pre + aux_start - 1
                        elif ind == 1:
                            locs[i] = pos + aux_start - 1
                        elif ind == 2:
                            locs[i] = aux_loc + aux_start - 1
                        i = i + 1

        #plt.scatter(locs,x[locs],marker = 'o',color = 'yellow', label = 'After not-peak correction')

        # Correction of peaks according to amplitude
        len_window = np.round(t*fs)
        #print('Window length: ' + str(len_window))
        keep = np.empty(0)
        first_i = 0
        second_i = len_window - 1
        while second_i < len(x):
            ind1, = np.where(locs >= first_i)
            ind2, = np.where(locs <= second_i)
            ind = np.intersect1d(ind1, ind2)
            win = locs[ind]
            if np.median(x[win]) > 0:
                th_amp_low = 0.5*np.median(x[win])
                th_amp_high = 3*np.median(x[win])
            else:
                th_amp_low = -3*np.median(x[win])
                th_amp_high = 1.5*np.median(x[win])
            ind1, = np.where(x[win] >= th_amp_low)
            ind2, = np.where(x[win] <= th_amp_high)
            aux_keep = np.intersect1d(ind1,ind2)
            keep = np.append(keep, aux_keep)

            first_i = second_i + 1
            second_i = second_i + stride*fs - 1

        if len(keep) != 0:
            keep = np.unique(keep)
            locs = locs[keep.astype(int)]

        #plt.scatter(locs,x[locs],marker = 'o',color = 'purple', label = 'After amplitude correction')
        #plt.legend()

        return locs
    
    def find_closest_peak(self, x, loc, dir_search):
        pos = -1
        if dir_search == 'backward':
            i = loc - 2
            while i > 0:
                if (x[i] > x[i - 1]) and (x[i] > x[i + 1]):
                    pos = i
                    i = 0
                else:
                    i = i - 1
            if pos == -1:
                pos = loc
        elif dir_search == 'forward':
            i = loc + 1
            while i < len(x) - 1:
                if (x[i] > x[i - 1]) and (x[i] > x[i + 1]):
                    pos = i
                    i = len(x)
                else:
                    i = i + 1
            if pos == -1:
                pos = loc

        return pos

    
    def d2max(self, x, fs):
        # Bandpass filter
        if len(x) < 4098:
            z_fill = np.zeros(4098 - len(x) + 1)
            x_z = np.append(x, z_fill)
        sos = sp.butter(10, [0.5, 8], btype = 'bp', analog = False, output = 'sos', fs = fs)
        x_f = sp.sosfiltfilt(sos, x_z)

        # Signal clipping
        ind, = np.where(x_f < 0)
        x_c = x_f
        x_c[ind] = 0

        # Signal squaring
        x_s = x_c**2

        #plt.figure()
        #plt.plot(x)
        #plt.plot(x_z)
        #plt.plot(x_f)
        #plt.plot(x_c)
        #plt.plot(x_s)

        # Blocks of interest
        w1 = (111e-3)*fs
        w1 = int(2*np.floor(w1/2) + 1)
        b = (1/w1)*np.ones(w1)
        ma_pk = sp.filtfilt(b,1,x_s)

        w2 = (667e-3)*fs
        w2 = int(2*np.floor(w2/2) + 1)
        b = (1/w2)*np.ones(w1)
        ma_bpm = sp.filtfilt(b,1,x_s)

        #plt.figure()
        #plt.plot(x_s/np.max(x_s))
        #plt.plot(ma_pk/np.max(ma_pk))
        #plt.plot(ma_bpm/np.max(ma_bpm))

        # Thresholding
        alpha = 0.02*np.mean(ma_pk)
        th_1 = ma_bpm + alpha
        th_2 = w1
        boi = (ma_pk > th_1).astype(int)

        blocks_init, = np.where(np.diff(boi) > 0)
        blocks_init = blocks_init + 1
        blocks_end, = np.where(np.diff(boi) < 0)
        blocks_end = blocks_end + 1

        if blocks_init[0] > blocks_end[0]:
            blocks_init = np.append(1, blocks_init)
        if blocks_init[-1] > blocks_end[-1]:
            blocks_end = np.append(blocks_end, len(x_s))

        #print('Initial locs BOI: ' + str(blocks_init))
        #print('Final locs BOI: ' + str(blocks_end))
        #plt.figure()
        #plt.plot(x_s[range(len(x))]/np.max(x_s))
        #plt.plot(boi[range(len(x))])

        # Search for peaks inside BOIs
        len_blks = np.zeros(len(blocks_init))
        ibis = np.zeros(len(blocks_init))
        for i in range(len(blocks_init)):
            ind, = np.where(blocks_end > blocks_init[i])
            ind = ind[0]
            len_blks[i] = blocks_end[ind] - blocks_init[i]
            if len_blks[i] >= th_2:
                aux = x[blocks_init[i]:blocks_end[ind]]
                if len(aux) != 0:
                    max_val = np.max(aux)
                    max_ind, = np.where(max_val == aux)
                    ibis[i] = max_ind + blocks_init[i] - 1

        ind, = np.where(len_blks < th_2)
        if len(ind) != 0:
            for i in range(len(ind)):
                boi[blocks_init[i]:blocks_end[i]] = 0
        ind, = np.where(ibis == 0)
        ibis = (np.delete(ibis, ind)).astype(int)

        #plt.plot(boi[range(len(x))])
        #plt.figure()
        #plt.plot(x)
        #plt.scatter(ibis, x[ibis], marker = 'o',color = 'red')

        return ibis

    def fiducial_points(self, x,pks,fs,vis):
        # First, second and third derivatives
        d1x = sp.savgol_filter(x, 9, 5, deriv = 1) 
        d2x = sp.savgol_filter(x, 9, 5, deriv = 2) 
        d3x = sp.savgol_filter(x, 9, 5, deriv = 3) 
    
        #plt.figure()
        #plt.plot(x/np.max(x))
        #plt.plot(d1x/np.max(d1x))
        #plt.plot(d2x/np.max(d2x))
        #plt.plot(d3x/np.max(d3x))
    
        # Search in time series: Onsets between consecutive peaks
        ons = np.empty(0)
        for i in range(len(pks) - 1):
            start = pks[i]
            stop = pks[i + 1]
            ibi = x[start:stop]
            #plt.figure()
            #plt.plot(ibi, color = 'black')
            aux_ons, = np.where(ibi == np.min(ibi))
            ind_ons = aux_ons.astype(int)
            ons = np.append(ons, ind_ons + start)   
            #plt.plot(ind_ons, ibi[ind_ons], marker = 'o', color = 'red') 
        ons = ons.astype(int)
        #print('Onsets: ' + str(ons))
        #plt.figure()
        #plt.plot(x, color = 'black')
        #plt.scatter(pks, x[pks], marker = 'o', color = 'red') 
        #plt.scatter(ons, x[ons], marker = 'o', color = 'blue') 
    
        # Search in time series: Diastolic peak and dicrotic notch between consecutive onsets
        dia = np.empty(0)
        dic = np.empty(0)
        for i in range(len(ons) - 1):
            start = ons[i]
            stop = ons[i + 1]
            ind_pks, = np.intersect1d(np.where(pks < stop), np.where(pks > start))
            ind_pks = pks[ind_pks]
            ibi_portion = x[ind_pks:stop]
            ibi_2d_portion = d2x[ind_pks:stop]
            #plt.figure()
            #plt.plot(ibi_portion/np.max(ibi_portion))
            #plt.plot(ibi_2d_portion/np.max(ibi_2d_portion))
            aux_dic, _ = sp.find_peaks(ibi_2d_portion)
            aux_dic = aux_dic.astype(int)
            aux_dia, _ = sp.find_peaks(-ibi_2d_portion)
            aux_dia = aux_dia.astype(int)   
            if len(aux_dic) != 0:
                ind_max, = np.where(ibi_2d_portion[aux_dic] == np.max(ibi_2d_portion[aux_dic]))
                aux_dic_max = aux_dic[ind_max]
                if len(aux_dia) != 0:
                    nearest = aux_dia - aux_dic_max
                    aux_dic = aux_dic_max
                    dic = np.append(dic, (aux_dic + ind_pks).astype(int))
                    #plt.scatter(aux_dic, ibi_portion[aux_dic]/np.max(ibi_portion), marker = 'o')
                    ind_dia, = np.where(nearest > 0)
                    aux_dia = aux_dia[ind_dia]
                    nearest = nearest[ind_dia]
                    if len(nearest) != 0:
                        ind_nearest, = np.where(nearest == np.min(nearest))
                        aux_dia = aux_dia[ind_nearest]
                        dia = np.append(dia, (aux_dia + ind_pks).astype(int))
                        #plt.scatter(aux_dia, ibi_portion[aux_dia]/np.max(ibi_portion), marker = 'o')
                        #break
                else:
                    dic = np.append(dic, (aux_dic_max + ind_pks).astype(int))
                    #plt.scatter(aux_dia, ibi_portion[aux_dia]/np.max(ibi_portion), marker = 'o')     
        dia = dia.astype(int)
        dic = dic.astype(int)
        #plt.scatter(dia, x[dia], marker = 'o', color = 'orange')
        #plt.scatter(dic, x[dic], marker = 'o', color = 'green')
    
        # Search in D1: Maximum slope point
        m1d = np.empty(0)
        for i in range(len(ons) - 1):
            start = ons[i]
            stop = ons[i + 1]
            ind_pks, = np.intersect1d(np.where(pks < stop), np.where(pks > start))
            ind_pks = pks[ind_pks]
            ibi_portion = x[start:ind_pks]
            ibi_1d_portion = d1x[start:ind_pks]
            #plt.figure()
            #plt.plot(ibi_portion/np.max(ibi_portion))
            #plt.plot(ibi_1d_portion/np.max(ibi_1d_portion))
            aux_m1d, _ = sp.find_peaks(ibi_1d_portion)
            aux_m1d = aux_m1d.astype(int)  
            if len(aux_m1d) != 0:
                ind_max, = np.where(ibi_1d_portion[aux_m1d] == np.max(ibi_1d_portion[aux_m1d]))
                aux_m1d_max = aux_m1d[ind_max]
                if len(aux_m1d_max) > 1:
                    aux_m1d_max = aux_m1d_max[0]
                m1d = np.append(m1d, (aux_m1d_max + start).astype(int))
                #plt.scatter(aux_m1d, ibi_portion[aux_dic]/np.max(ibi_portion), marker = 'o')
                #break    
        m1d = m1d.astype(int)
        #plt.scatter(m1d, x[m1d], marker = 'o', color = 'purple')
    
        # Search in time series: Tangent intersection points
        tip = np.empty(0)
        for i in range(len(ons) - 1):
            start = ons[i]
            stop = ons[i + 1]
            ibi_portion = x[start:stop]
            ibi_1d_portion = d1x[start:stop]
            ind_m1d, = np.intersect1d(np.where(m1d < stop), np.where(m1d > start))
            ind_m1d = m1d[ind_m1d] - start
            #plt.figure()
            #plt.plot(ibi_portion/np.max(ibi_portion))
            #plt.plot(ibi_1d_portion/np.max(ibi_1d_portion))
            #plt.scatter(ind_m1d, ibi_portion[ind_m1d]/np.max(ibi_portion), marker = 'o')
            #plt.scatter(ind_m1d, ibi_1d_portion[ind_m1d]/np.max(ibi_1d_portion), marker = 'o')
            aux_tip = np.round(((ibi_portion[0] - ibi_portion[ind_m1d])/ibi_1d_portion[ind_m1d]) + ind_m1d)
            aux_tip = aux_tip.astype(int)
            tip = np.append(tip, (aux_tip + start).astype(int))        
            #plt.scatter(aux_tip, ibi_portion[aux_tip]/np.max(ibi_portion), marker = 'o')
            #break
        tip = tip.astype(int)
        #plt.scatter(tip, x[tip], marker = 'o', color = 'aqua')
    
        # Search in D2: A, B, C, D and E points
        a2d = np.empty(0)
        b2d = np.empty(0)
        c2d = np.empty(0)
        d2d = np.empty(0)
        e2d = np.empty(0)
        for i in range(len(ons) - 1):
            start = ons[i]
            stop = ons[i + 1]
            ibi_portion = x[start:stop]
            ibi_1d_portion = d1x[start:stop]
            ibi_2d_portion = d2x[start:stop]
            ind_m1d = np.intersect1d(np.where(m1d > start),np.where(m1d < stop))
            ind_m1d = m1d[ind_m1d]
            #plt.figure()
            #plt.plot(ibi_portion/np.max(ibi_portion))
            #plt.plot(ibi_1d_portion/np.max(ibi_1d_portion))
            #plt.plot(ibi_2d_portion/np.max(ibi_2d_portion))
            aux_m2d_pks, _ = sp.find_peaks(ibi_2d_portion)
            aux_m2d_ons, _ = sp.find_peaks(-ibi_2d_portion)
            # a point:
            ind_a, = np.where(ibi_2d_portion[aux_m2d_pks] == np.max(ibi_2d_portion[aux_m2d_pks]))
            ind_a = aux_m2d_pks[ind_a]
            if (ind_a < ind_m1d):
                a2d = np.append(a2d, ind_a + start)
                #plt.scatter(ind_a, ibi_2d_portion[ind_a]/np.max(ibi_2d_portion), marker = 'o')
                # b point:
                ind_b = np.where(ibi_2d_portion[aux_m2d_ons] == np.min(ibi_2d_portion[aux_m2d_ons]))
                ind_b = aux_m2d_ons[ind_b]
                if (ind_b > ind_a) and (ind_b < len(ibi_2d_portion)):
                    b2d = np.append(b2d, ind_b + start)
                    #plt.scatter(ind_b, ibi_2d_portion[ind_b]/np.max(ibi_2d_portion), marker = 'o')
            # e point:
            ind_e, = np.where(aux_m2d_pks > ind_m1d - start)
            aux_m2d_pks = aux_m2d_pks[ind_e]
            ind_e, = np.where(aux_m2d_pks < 0.6*len(ibi_2d_portion))
            ind_e = aux_m2d_pks[ind_e]
            if len(ind_e) >= 1:
                if len(ind_e) >= 2:
                    ind_e = ind_e[1]
                e2d = np.append(e2d, ind_e + start)
                #plt.scatter(ind_e, ibi_2d_portion[ind_e]/np.max(ibi_2d_portion), marker = 'o')
                # c point:
                ind_c, = np.where(aux_m2d_pks < ind_e)
                if len(ind_c) != 0:
                    ind_c_aux = aux_m2d_pks[ind_c]
                    ind_c, = np.where(ibi_2d_portion[ind_c_aux] == np.max(ibi_2d_portion[ind_c_aux]))
                    ind_c = ind_c_aux[ind_c]
                    if len(ind_c) != 0:
                        c2d = np.append(c2d, ind_c + start)
                        #plt.scatter(ind_c, ibi_2d_portion[ind_c]/np.max(ibi_2d_portion), marker = 'o')
                else:
                    aux_m1d_ons, _ = sp.find_peaks(-ibi_1d_portion)
                    ind_c, = np.where(aux_m1d_ons < ind_e)
                    ind_c_aux = aux_m1d_ons[ind_c]
                    if len(ind_c) != 0:
                        ind_c, = np.where(ind_c_aux > ind_b)
                        ind_c = ind_c_aux[ind_c]
                        if len(ind_c) > 1:
                            ind_c = ind_c[0]
                        c2d = np.append(c2d, ind_c + start)
                        #plt.scatter(ind_c, ibi_2d_portion[ind_c]/np.max(ibi_2d_portion), marker = 'o')
                # d point:
                if len(ind_c) != 0:
                    ind_d = np.intersect1d(np.where(aux_m2d_ons < ind_e), np.where(aux_m2d_ons > ind_c))
                    if len(ind_d) != 0:
                        ind_d_aux = aux_m2d_ons[ind_d]
                        ind_d, = np.where(ibi_2d_portion[ind_d_aux] == np.min(ibi_2d_portion[ind_d_aux]))
                        ind_d = ind_d_aux[ind_d]
                        if len(ind_d) != 0:
                            d2d = np.append(d2d, ind_d + start)
                            #plt.scatter(ind_d, ibi_2d_portion[ind_d]/np.max(ibi_2d_portion), marker = 'o')                
                    else:
                        ind_d = ind_c
                        d2d = np.append(d2d, ind_d + start)
                        #plt.scatter(ind_d, ibi_2d_portion[ind_d]/np.max(ibi_2d_portion), marker = 'o')
        a2d = a2d.astype(int)
        b2d = b2d.astype(int)
        c2d = c2d.astype(int)
        d2d = d2d.astype(int)
        e2d = e2d.astype(int)
        #plt.figure()
        #plt.plot(d2x, color = 'black')
        #plt.scatter(a2d, d2x[a2d], marker = 'o', color = 'red') 
        #plt.scatter(b2d, d2x[b2d], marker = 'o', color = 'blue')
        #plt.scatter(c2d, d2x[c2d], marker = 'o', color = 'green')
        #plt.scatter(d2d, d2x[d2d], marker = 'o', color = 'orange')
        #plt.scatter(e2d, d2x[e2d], marker = 'o', color = 'purple')
    
        # Search in D3: P1 and P2 points
        p1p = np.empty(0)
        p2p = np.empty(0)
        for i in range(len(ons) - 1):
            start = ons[i]
            stop = ons[i + 1]
            ibi_portion = x[start:stop]
            ibi_1d_portion = d1x[start:stop]
            ibi_2d_portion = d2x[start:stop]
            ibi_3d_portion = d3x[start:stop]
            ind_b = np.intersect1d(np.where(b2d > start),np.where(b2d < stop))
            ind_b = b2d[ind_b]
            ind_c = np.intersect1d(np.where(c2d > start),np.where(c2d < stop))
            ind_c = c2d[ind_c]
            ind_d = np.intersect1d(np.where(d2d > start),np.where(d2d < stop))
            ind_d = d2d[ind_d]
            ind_dic = np.intersect1d(np.where(dic > start),np.where(dic < stop))
            ind_dic = dic[ind_dic]
            #plt.figure()
            #plt.plot(ibi_portion/np.max(ibi_portion))
            #plt.plot(ibi_1d_portion/np.max(ibi_1d_portion))
            #plt.plot(ibi_2d_portion/np.max(ibi_2d_portion))
            #plt.plot(ibi_3d_portion/np.max(ibi_3d_portion))
            #plt.scatter(ind_b - start, ibi_3d_portion[ind_b - start]/np.max(ibi_3d_portion), marker = 'o')
            #plt.scatter(ind_c - start, ibi_3d_portion[ind_c - start]/np.max(ibi_3d_portion), marker = 'o')
            #plt.scatter(ind_d - start, ibi_3d_portion[ind_d - start]/np.max(ibi_3d_portion), marker = 'o')
            #plt.scatter(ind_dic - start, ibi_3d_portion[ind_dic - start]/np.max(ibi_3d_portion), marker = 'o')
            aux_p3d_pks, _ = sp.find_peaks(ibi_3d_portion)
            aux_p3d_ons, _ = sp.find_peaks(-ibi_3d_portion)
            # P1:
            if (len(aux_p3d_pks) != 0 and len(ind_b) != 0):
                ind_p1, = np.where(aux_p3d_pks > ind_b - start)
                if len(ind_p1) != 0:
                    ind_p1 = aux_p3d_pks[ind_p1[0]]
                    p1p = np.append(p1p, ind_p1 + start)
                    #plt.scatter(ind_p1, ibi_3d_portion[ind_p1]/np.max(ibi_3d_portion), marker = 'o')
            # P2:
            if (len(aux_p3d_ons) != 0 and len(ind_c) != 0 and len(ind_d) != 0):
                if ind_c == ind_d:
                    ind_p2, = np.where(aux_p3d_ons > ind_d - start)
                    ind_p2 = aux_p3d_ons[ind_p2[0]]
                else:
                    ind_p2, = np.where(aux_p3d_ons < ind_d - start)
                    ind_p2 = aux_p3d_ons[ind_p2[-1]]
                if len(ind_dic) != 0:
                    aux_x_pks, _ = sp.find_peaks(ibi_portion)
                    if ind_p2 > ind_dic - start:
                        ind_between = np.intersect1d(np.where(aux_x_pks < ind_p2), np.where(aux_x_pks > ind_dic - start))
                    else:
                        ind_between = np.intersect1d(np.where(aux_x_pks > ind_p2), np.where(aux_x_pks < ind_dic - start))
                    if len(ind_between) != 0:
                        ind_p2 = aux_x_pks[ind_between[0]]
                p2p = np.append(p2p, ind_p2 + start)
                #plt.scatter(ind_p2, ibi_3d_portion[ind_p2]/np.max(ibi_3d_portion), marker = 'o')
        p1p = p1p.astype(int)
        p2p = p2p.astype(int)
        #plt.figure()
        #plt.plot(d3x, color = 'black')
        #plt.scatter(p1p, d3x[p1p], marker = 'o', color = 'green') 
        #plt.scatter(p2p, d3x[p2p], marker = 'o', color = 'orange')
    
        # Added by PC: Magnitudes of second derivative points
        bmag2d = np.zeros(len(b2d))
        cmag2d = np.zeros(len(b2d))
        dmag2d = np.zeros(len(b2d))
        emag2d = np.zeros(len(b2d))
        for beat_no in range(0,len(d2d)):
            bmag2d[beat_no] = d2x[b2d[beat_no]]/d2x[a2d[beat_no]]
            cmag2d[beat_no] = d2x[c2d[beat_no]]/d2x[a2d[beat_no]]
            dmag2d[beat_no] = d2x[d2d[beat_no]]/d2x[a2d[beat_no]]        
            emag2d[beat_no] = d2x[e2d[beat_no]]/d2x[a2d[beat_no]]    
        
        # Added by PC: Refine the list of fiducial points to only include those corresponding to beats for which a full set of points is available
        off = ons[1:]
        ons = ons[:-1]
        if pks[0] < ons[0]:
            pks = pks[1:]
        if pks[-1] > off[-1]:
            pks = pks[:-1]
    
        # Visualise results
        if vis == True:
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex = True, sharey = False, figsize=(10,10))
            fig.suptitle('Fiducial points') 

            ax1.plot(x, color = 'black')
            ax1.scatter(pks, x[pks.astype(int)], color = 'orange', label = 'pks')
            ax1.scatter(ons, x[ons.astype(int)], color = 'green', label = 'ons')
            ax1.scatter(off, x[off.astype(int)], marker = '*', color = 'green', label = 'off')
            ax1.scatter(dia, x[dia.astype(int)], color = 'yellow', label = 'dia')
            ax1.scatter(dic, x[dic.astype(int)], color = 'blue', label = 'dic')
            ax1.scatter(tip, x[tip.astype(int)], color = 'purple', label = 'dic')
            ax1.legend()
            ax1.set_ylabel('x')

            ax2.plot(d1x, color = 'black')
            ax2.scatter(m1d, d1x[m1d.astype(int)], color = 'orange', label = 'm1d')
            ax2.legend()
            ax2.set_ylabel('d1x')

            ax3.plot(d2x, color = 'black')
            ax3.scatter(a2d, d2x[a2d.astype(int)], color = 'orange', label = 'a')
            ax3.scatter(b2d, d2x[b2d.astype(int)], color = 'green', label = 'b')
            ax3.scatter(c2d, d2x[c2d.astype(int)], color = 'yellow', label = 'c')
            ax3.scatter(d2d, d2x[d2d.astype(int)], color = 'blue', label = 'd')
            ax3.scatter(e2d, d2x[e2d.astype(int)], color = 'purple', label = 'e')
            ax3.legend()
            ax3.set_ylabel('d2x')

            ax4.plot(d3x, color = 'black')
            ax4.scatter(p1p, d3x[p1p.astype(int)], color = 'orange', label = 'p1')
            ax4.scatter(p2p, d3x[p2p.astype(int)], color = 'green', label = 'p2')
            ax4.legend()
            ax4.set_ylabel('d3x')

            plt.subplots_adjust(left = 0.1,
                                bottom = 0.1, 
                                right = 0.9, 
                                top = 0.9, 
                                wspace = 0.4, 
                                hspace = 0.4)
        
        # Creation of dictionary
        fidp = {'pks': pks.astype(int),
                'ons': ons.astype(int),
                'off': off.astype(int),  # Added by PC
                'tip': tip.astype(int),
                'dia': dia.astype(int),
                'dic': dic.astype(int),
                'm1d': m1d.astype(int),
                'a2d': a2d.astype(int),
                'b2d': b2d.astype(int),
                'c2d': c2d.astype(int),
                'd2d': d2d.astype(int),
                'e2d': e2d.astype(int),
                'bmag2d': bmag2d,
                'cmag2d': cmag2d,
                'dmag2d': dmag2d,
                'emag2d': emag2d,
                'p1p': p1p.astype(int),
                'p2p': p2p.astype(int)
                }
    
        return fidp
    
    def detect_beats(self):
        temp_fs = 125
        ibis = self.pulse_detect(self.ppg_filt, temp_fs, 5)
        print(ibis)

        if self.gr:
            ax3 = plt.subplot(323)
            t = np.arange(0,len(self.ppg_filt)/self.fs,1.0/self.fs)
            ax3.plot(t, self.ppg_filt, color = 'black')
            ax3.scatter(t[0] + ibis/self.fs, self.ppg_filt[ibis], color = 'orange', marker = 'o')
            ax3.axis("tight")
            ax3.set_title("IBIs detection")

        self.fidp = self.fiducial_points(self.ppg_filt, ibis, self.fs, vis = True)

    def calc_features(self):
        delta_t = np.zeros(len(self.fidp["dia"]))
        for beat_no in range(len(self.fidp["dia"])):
                delta_t[beat_no] = (self.fidp["dia"][beat_no]-self.fidp["pks"][beat_no])/self.fs
        print("Values of Delta T:")
        print(delta_t)

        agi = np.zeros(len(self.fidp["dia"]))
        for beat_no in range(len(self.fidp["dia"])):
            agi[beat_no] = (self.fidp["bmag2d"][beat_no]-self.fidp["cmag2d"][beat_no]-self.fidp["dmag2d"][beat_no]-self.fidp["emag2d"][beat_no])/self.fs
        print("Values of Aging Index:")
        print(agi)

    def run(self):
        print(self.ppg)
        self.plot_raw_ppg()
        self.filter_ppg()
        self.detect_beats()
        self.calc_features()

        # show results
        plt.show()
        return [self.min_max_amp]
