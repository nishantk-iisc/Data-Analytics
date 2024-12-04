
import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd
import line_fit as lf
import gamma_fit as gd


if __name__ == '__main__':
    print('Visual Neuroscience!')
    fire_rate_data = pd.read_csv('../data/02_data_visual_neuroscience_firingrates.csv')
    srch_time_data = pd.read_csv('../data/02_data_visual_neuroscience_searchtimes.csv')

    # Fit straight line
    line_fit_ = lf.LineFit(srch_time_data, fire_rate_data)
    avg_search_times = line_fit_.find_avg_search_time()
    entropy_relative_data, l1_distance_data = line_fit_.calc_entropy_and_l1_dist()
    line_fit_.plot_srch_vs_l1_dist()
    line_fit_.plot_srch_vs_entropy()
    amGmSearchEntropyRatio, amGmSearchL1DistanceRation = line_fit_.calc_am_gm_spread()

    # Fit Gamma Distribution
    gd_fit = gd.Gamma_Dist_Fitter(srch_time_data)
    gd_fit.select_grps_randomly()
    gd_fit.mean_sd_rand_grps()
    gd_fit.mean_vs_sd_plot()
    gd_fit.find_shape_para_values()
    gd_fit.find_rate_para_and_kolmogorov_stat()
