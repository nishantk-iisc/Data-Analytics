
import numpy as np
import utility
from scipy.stats.mstats import gmean
from matplotlib import pyplot as plt


class LineFit:

    def __init__(self, srch_time_data, fire_rate_data):
        self._srch_time_data = srch_time_data
        self._fire_rate_data = fire_rate_data
        self._avg_search_times = []
        self._entropy_relative_data = []
        self._l1_dist_data = []
        self._inv_srch_time = []
        self._amGmSearchEntropyRatio = None
        self._amGmSearchL1DistanceRation = None

    def find_avg_search_time(self):
        len_search_data = self._srch_time_data.shape[1]
        print('Size of search data list :: ' + str(len_search_data))
        for i in range(len_search_data):
            search_time_col_i = np.array(self._srch_time_data.values[:, i][2:]).astype(np.float64)
            # print(search_time_col_i[2:])
            avg_search_time = utility.get_avg_search_times(search_time_col_i, 328)
            self._avg_search_times.append(avg_search_time)
            self._inv_srch_time = [1000 / search_time for search_time in self._avg_search_times]
        return self._avg_search_times

    def calc_entropy_and_l1_dist(self):
        set_count = 4
        col_cnt_per_set = 6
        for i in range(set_count):
            if i != 3:
                for j in range(col_cnt_per_set // 2):
                    col_idx = i * col_cnt_per_set + 2 * j
                    # print(col_idx)
                    f_rate_0 = np.array(self._fire_rate_data.values[:, col_idx][2:]).astype(np.float64)
                    f_rate_1 = np.array(self._fire_rate_data.values[:, col_idx + 1][2:]).astype(np.float64)
                    ij_relative_entropy = utility.get_entorpy(f_rate_0, f_rate_1)
                    self._entropy_relative_data.append(ij_relative_entropy)
                    ij_l1_distance = utility.get_dist_l1(f_rate_0, f_rate_1)
                    self._l1_dist_data.append(ij_l1_distance)

                    ji_relative_entropy = utility.get_entorpy(f_rate_1, f_rate_0)
                    self._entropy_relative_data.append(ji_relative_entropy)
                    ji_l1_distance = utility.get_dist_l1(f_rate_1, f_rate_0)
                    self._l1_dist_data.append(ji_l1_distance)
            else:
                for j in range(3):
                    col_idx = i * col_cnt_per_set + 2 * j
                    f_rate_0 = np.array(self._fire_rate_data.values[:, col_idx][2:]).astype(np.float64)
                    f_rate_1 = np.array(self._fire_rate_data.values[:, col_idx + 2][2:]).astype(np.float64)
                    ij_relative_entropy_1 = utility.get_entorpy(f_rate_0, f_rate_1)
                    ij_l1_distance_1 = utility.get_dist_l1(f_rate_0, f_rate_1)
                    ji_relative_entropy_1 = utility.get_entorpy(f_rate_1, f_rate_0)
                    ji_l1_distance_1 = utility.get_dist_l1(f_rate_1, f_rate_0)

                    f_rate_0 = np.array(self._fire_rate_data.values[:, col_idx][2:]).astype(np.float64)
                    f_rate_1 = np.array(self._fire_rate_data.values[:, col_idx + 3][2:]).astype(np.float64)
                    ij_relative_entropy_2 = utility.get_entorpy(f_rate_0, f_rate_1)
                    ij_l1_distance_2 = utility.get_dist_l1(f_rate_0, f_rate_1)
                    ji_relative_entropy_2 = utility.get_entorpy(f_rate_1, f_rate_0)
                    ji_l1_distance_2 = utility.get_dist_l1(f_rate_1, f_rate_0)

                    f_rate_0 = np.array(self._fire_rate_data.values[:, col_idx + 1][2:]).astype(np.float64)
                    f_rate_1 = np.array(self._fire_rate_data.values[:, col_idx + 2][2:]).astype(np.float64)
                    ij_relative_entropy_3 = utility.get_entorpy(f_rate_0, f_rate_1)
                    ij_l1_distance_3 = utility.get_dist_l1(f_rate_0, f_rate_1)
                    ji_relative_entropy_3 = utility.get_entorpy(f_rate_1, f_rate_0)
                    ji_l1_distance_3 = utility.get_dist_l1(f_rate_1, f_rate_0)

                    f_rate_0 = np.array(self._fire_rate_data.values[:, col_idx + 1][2:]).astype(np.float64)
                    f_rate_1 = np.array(self._fire_rate_data.values[:, col_idx + 3][2:]).astype(np.float64)
                    ij_relative_entropy_4 = utility.get_entorpy(f_rate_0, f_rate_1)
                    ij_l1_distance_4 = utility.get_dist_l1(f_rate_0, f_rate_1)
                    ji_relative_entropy_4 = utility.get_entorpy(f_rate_1, f_rate_0)
                    ji_l1_distance_4 = utility.get_dist_l1(f_rate_1, f_rate_0)

                    ij_relative_entropy = np.mean([ij_relative_entropy_1, ij_relative_entropy_2, ij_relative_entropy_3,
                                                   ij_relative_entropy_4])
                    ij_l1_distance = np.mean([ij_l1_distance_1, ij_l1_distance_2, ij_l1_distance_3, ij_l1_distance_4])
                    self._entropy_relative_data.append(ij_relative_entropy)
                    self._l1_dist_data.append(ij_l1_distance)

                    ji_relative_entropy = np.mean([ji_relative_entropy_1, ji_relative_entropy_2, ji_relative_entropy_3,
                                                   ji_relative_entropy_4])
                    ji_l1_distance = np.mean([ji_l1_distance_1, ji_l1_distance_2, ji_l1_distance_3, ji_l1_distance_4])
                    self._entropy_relative_data.append(ji_relative_entropy)
                    self._l1_dist_data.append(ji_l1_distance)

        print('Size of relative entropy list :: ' + str(len(self._entropy_relative_data)))
        # print(self._entropy_relative_data)
        print('Size of L1 distance list :: ' + str(len(self._l1_dist_data)))
        # print(self._l1_dist_data)
        return self._entropy_relative_data, self._l1_dist_data

    @staticmethod
    def _fit_straight_line_through_origin(x_, y_):
        x_ = x_[:,np.newaxis]
        _a, _residuals, _, _ = np.linalg.lstsq(x_, y_, rcond=None)
        return _a, _residuals

    def plot_srch_vs_entropy(self):
        ax = plt.subplot(111)
        plt.xlabel('Relative Entropy distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        ax.scatter(self._entropy_relative_data, self._inv_srch_time, c='red')
        slope_, residual_error_ = LineFit._fit_straight_line_through_origin(np.array(self._entropy_relative_data),
                                                             np.array(self._inv_srch_time))
        print('Slope for relative entropy vs. inverse search time curve :: ', slope_[0])
        print('Residual error for the straight line fit for relative entropy :: ', residual_error_[0])
        ax.plot(self._entropy_relative_data, slope_*self._entropy_relative_data)
        plt.savefig('../plots/relative_entropy.png')
        plt.close()
        # plt.show()

    def plot_srch_vs_l1_dist(self):
        ax = plt.subplot(111)
        plt.xlabel('L1 distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        ax.scatter(self._l1_dist_data, self._inv_srch_time, c='red')
        slope_, residual_error_ = LineFit._fit_straight_line_through_origin(np.array(self._l1_dist_data),
                                                             np.array(self._inv_srch_time))
        print('Slope for l1 distance vs. inverse search time curve :: ', slope_[0])
        print('Residual error for the straight line fit for l1 distance :: ', residual_error_[0])
        ax.plot(self._l1_dist_data, slope_ * self._l1_dist_data)
        plt.savefig('../plots/l1_distance_plot.png')
        plt.close()

    def calc_am_gm_spread(self):
        search_entropy_product = np.multiply(self._avg_search_times, self._entropy_relative_data)
        search_l1_distance_product = np.multiply(self._avg_search_times, self._l1_dist_data)

        AmProductSearchEntropy = np.mean(search_entropy_product)
        GmProductSearchEntropy = gmean(search_entropy_product)
        print('Arithmetic mean for search * relative entropy :: ' + str(AmProductSearchEntropy))
        print('Geometric mean for search * relative entropy :: ' + str(GmProductSearchEntropy))

        AmProductSearchL1Distance = np.mean(search_l1_distance_product)
        GmProductSearchL1Distance = gmean(search_l1_distance_product)
        print('Arithmetic mean for search * L1 distance :: ' + str(AmProductSearchL1Distance))
        print('Geometric mean for search * L1 distance :: ' + str(GmProductSearchL1Distance))

        self._amGmSearchEntropyRatio = AmProductSearchEntropy / GmProductSearchEntropy
        print('Ratio of AM and GM for search * relative entropy :: ' + str(self._amGmSearchEntropyRatio))

        self._amGmSearchL1DistanceRation = AmProductSearchL1Distance / GmProductSearchL1Distance
        print('Ratio of AM and GM for search * L1 distance :: ' + str(self._amGmSearchL1DistanceRation))

        return self._amGmSearchEntropyRatio, self._amGmSearchL1DistanceRation
