
import math
import numpy as np

def get_entorpy(f_rate0, f_rate1):
    par_m = 24
    par_lambda = 250
    bias = 1/(2*par_m*par_lambda)
    entropy_relative = []
    assert len(f_rate0) == len(f_rate1)
    count_valid_f_rate = 0
    for i in range(len(f_rate0)):
        if np.isnan(f_rate0[i]) or np.isnan(f_rate1[i]):
            pass
        else:
            if f_rate0[i] > bias:
                entropy_relative_neu_i = f_rate0[i]*math.log((f_rate0[i] - bias)/(f_rate1[i] + bias)) - f_rate0[i] + f_rate1[i]
                entropy_relative.append(entropy_relative_neu_i)
            else:
                entropy_relative.append(f_rate1[i])
            count_valid_f_rate += 1
    return np.sum(entropy_relative)/count_valid_f_rate


def get_dist_l1(f_rate0, f_rate1):
    dist_l1 = []
    assert len(f_rate0) == len(f_rate1)
    count_valid_f_rate = 0
    for i in range(len(f_rate0)):
        if np.isnan(f_rate0[i]) or np.isnan(f_rate1[i]):
            pass
        else:
            dist_l1_neuron_i = abs(f_rate0[i] - f_rate1[i])
            dist_l1.append(dist_l1_neuron_i)
        count_valid_f_rate += 1

    return np.sum(dist_l1)/len(f_rate1)


def get_avg_search_times(react_times, base_line):
    return np.nanmean(react_times) - base_line
