
import random
import numpy as np
import math
from matplotlib import pyplot as plt
from random import shuffle
import itertools
from scipy import stats


class Gamma_Dist_Fitter:

    def __init__(self, look_up_time_data):
        self._look_up_time_data = look_up_time_data
        self._grps_count = look_up_time_data.shape[1]
        self._rand_grps = None
        self._nors_mean_list = []
        self._nor_stddev_list = []
        self._shape_val = None
        self._rate_val = None

    def select_grps_randomly(self):
        random_list = []
        while len(random_list) < self._grps_count//2:
            random_number = random.randint(0, self._grps_count-1)
            if random_number not in random_list:
                random_list.append(random_number)
        print('Random Numbers group :: ', random_list)
        self._rand_grps = random_list

    def mean_sd_rand_grps(self):
        for idx in self._rand_grps:
            srch_time_coli = np.array(self._look_up_time_data.values[:, idx][2:]).astype(np.float64)
            mean_ = np.nanmean(srch_time_coli)
            stddev_ = math.sqrt(np.nanvar(srch_time_coli))
            self._nors_mean_list.append(mean_)
            self._nor_stddev_list.append(stddev_)
        print('Means :: ', self._nors_mean_list)
        print('Standard Deviations :: ', self._nor_stddev_list)

    def mean_vs_sd_plot(self):
        ax = plt.subplot(111)
        plt.xlabel('Mean')
        plt.ylabel('Standard Deviation (sd)')
        ax.scatter(self._nors_mean_list, self._nor_stddev_list, color='r')
        plt.savefig('../plots/gamma_sd_mean.png')
        print("Plot Gamma std dev is saved.\n")
        plt.close()

    def find_shape_para_values(self):
        para_line = np.polyfit(self._nors_mean_list, self._nor_stddev_list, deg=1, full=True)
        self._shape_val = 1/math.pow(para_line[0][0], 2)
        print('Shape parameter :: ', self._shape_val)

    def find_rate_para_and_kolmogorov_stat(self):
        groups_left_outs = []
        nors_mean_list = []
        var_value_list = []
        cdf_values = []
        for idx in range(self._grps_count):
            if idx not in self._rand_grps:
                groups_left_outs.append(idx)

        for idx in groups_left_outs:
            srch_time_coli = np.array(self._look_up_time_data.values[:, idx][2:]).astype(np.float64)
            clean_srch_time_coli = [time for time in srch_time_coli if str(time) != 'nan']
            shuffle(clean_srch_time_coli)
            rdmized_search_times = clean_srch_time_coli[0:len(clean_srch_time_coli)//2]
            cdf_values.append(clean_srch_time_coli[len(clean_srch_time_coli)//2:len(clean_srch_time_coli)])
            mean_ = np.nanmean(rdmized_search_times)
            variance_ = np.nanvar(rdmized_search_times)
            nors_mean_list.append(mean_)
            var_value_list.append(variance_)


        para_line = np.polyfit(nors_mean_list, var_value_list, deg=1, full=True)
        self._rate_val = 1/para_line[0][0]
        print('Rate parameter :: ', self._rate_val)

        cdf_values = list(itertools.chain.from_iterable(cdf_values))
        sorted_cdf_value = np.sort(cdf_values)
        
        # Plot empirical gamma distribution
        y_cdf_emp_value = np.arange(len(sorted_cdf_value)) / float(len(sorted_cdf_value) - 1)
        plt.plot(sorted_cdf_value, y_cdf_emp_value)

        # # Plot gamma distribution
        x_gamma_val = np.linspace(0, sorted_cdf_value[-1], 200)
        y_gamma_val = stats.gamma.cdf(x_gamma_val, a=self._shape_val, scale=1/self._rate_val)
        plt.plot(x_gamma_val, y_gamma_val, color='r')
        plt.savefig('../plots/gamma_dist.png')
        plt.close()

        y_pdf = stats.gamma.rvs(size=len(cdf_values), a=self._shape_val, scale=1 / self._rate_val)
        kst_test = stats.ks_2samp(sorted_cdf_value, y_pdf)
        print('Kolmogorov statistic :: ', kst_test)
