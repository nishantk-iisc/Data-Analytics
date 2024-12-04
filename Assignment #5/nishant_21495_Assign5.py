#!/usr/bin/env python
# coding: utf-8
# author: Nishant Kumar

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import date
import datetime

def preprocessing():
    new_df = pd.read_csv('../COVID19_data.csv')
    # rename the columns
    new_df.columns = ['date', 'confirmed', 'recovered', 'deceased', 'other', 'tested', 'first', 'second', 'total']
    # drop the unused columns
    new_df.drop(['deceased', 'other', 'recovered', 'second', 'total'], axis = 1, inplace = True)
    new_df['date'] = pd.to_datetime(new_df['date'])
    full_df = new_df[new_df.date >= '2021-03-08'].copy()
    new_df = new_df[(new_df.date >= '2021-03-08') & (new_df.date <= '2021-04-26')].copy()
    new_df = new_df.reset_index().drop('index', axis = 1)
    new_df['confirmed'] = new_df['confirmed'].diff()
    new_df['tested'] = new_df['tested'].diff()
    new_df = new_df.drop(0, axis = 0).reset_index().drop('index', axis = 1)
    new_df['confirmed'] = new_df['confirmed'].astype(int)
    new_df['tested'] = new_df['tested'].astype(int)
    new_df['first'] = new_df['first'].astype(int)
    return new_df, full_df

def confirmedAvgCases(new_df, full_df, lt_date):
    ts_df = new_df.copy().to_numpy()
    cnf_avg = np.zeros(42)
    for day in range(42):
        for prev_day in range(7):
            cnf_avg[day] += ts_df[day + prev_day + 1][1] / 7 

    tst_data = full_df[['date', 'tested']].copy().to_numpy()
    for i in range(len(tst_data) - 1, 7, -1):
        tst_data[i][1] = (tst_data[i][1] - tst_data[i-7][1]) / 7
    tst_data = tst_data[8:] # starting ts_df from 16th March 2021
    # extrapolating the "tst_data" new_df till 31st Dec 2021
    while (lt_date - tst_data[-1][0]).days != 0:
        tst_data = np.append(tst_data, [[tst_data[-1][0] + datetime.timedelta(days = 1), tst_data[-1][1]]], axis = 0)

    ## creating ts_df (ground truth) for "first dose average"
    ft_df = new_df[['date', 'first']].copy().to_numpy()
    for i in range(len(ft_df) - 1, 6, -1):
        ft_df[i][1] = (ft_df[i][1] - ft_df[i-7][1]) / 7
    ft_df = ft_df[7:]
    # extrapolating the "ft_df dose" new_df till 31st Dec 2021
    while (lt_date - ft_df[-1][0]).days != 0:
        ft_df = np.append(ft_df, [[ft_df[-1][0] + datetime.timedelta(days = 1), 200000]], axis = 0)

    ## creating ts_df (ground truth) for "confirmed cases average" till 20th September (for Question 4)
    grnd_truth = full_df['confirmed'].copy().to_numpy()
    for i in range(len(grnd_truth) - 1, 7, -1):
        grnd_truth[i] = (grnd_truth[i] - grnd_truth[i-7]) / 7
    grnd_truth = grnd_truth[8:]

    return ft_df, tst_data, cnf_avg, grnd_truth

def generate_time_series(BETA, S0, E0, I0, R0, CIR0, V, tst_data, nors_days, N, ALPHA, GAMMA, EPS, waning = True):
    S = np.zeros(nors_days)
    E = np.zeros(nors_days)
    I = np.zeros(nors_days)
    R = np.zeros(nors_days)
    e = np.zeros(nors_days)

    S[0] = S0 
    E[0] = E0
    I[0] = I0 
    R[0] = R0

    st_date = datetime.datetime(2021, 3, 16)
    end_date = datetime.datetime(2021, 4, 26)
    waning_date = datetime.datetime(2021, 9, 11)
    lt_date = datetime.datetime(2021, 12, 31)

    for day in range(nors_days - 1):
        if day <= 30:
            delta_W = R0 / 30
        elif day >= 180:
            if waning == True:
                delta_W = R[day - 180] + EPS * V[day - 180][1]
            else:
                delta_W = 0
            pass
        else:
            delta_W = 0
        S[day + 1] = S[day] - BETA * S[day] * I[day] / N - EPS * V[day][1] + delta_W
        E[day + 1] = E[day] + BETA * S[day] * I[day] / N - ALPHA * E[day]
        I[day + 1] = I[day] + ALPHA * E[day] - GAMMA * I[day]
        R[day + 1] = R[day] + GAMMA * I[day] + EPS * V[day][1] - delta_W

    avg_S_val = np.zeros(nors_days)
    avg_E_val = np.zeros(nors_days)
    avg_I_val = np.zeros(nors_days)
    avg_R_val = np.zeros(nors_days)

    for day in range(nors_days):
        cnt_avg_val = 0
        for prev_day in range(day, day - 7, -1):
            if prev_day >= 0:
                avg_S_val[day] += S[prev_day]
                avg_E_val[day] += E[prev_day]
                avg_I_val[day] += I[prev_day]
                avg_R_val[day] += R[prev_day]
                cnt_avg_val += 1
        avg_S_val[day] = avg_S_val[day] / cnt_avg_val
        avg_E_val[day] = avg_E_val[day] / cnt_avg_val
        avg_I_val[day] = avg_I_val[day] / cnt_avg_val
        avg_R_val[day] = avg_R_val[day] / cnt_avg_val

    for day in range(nors_days):
        CIR = CIR0 * tst_data[0][1] / tst_data[day][1] 
        e[day] = avg_E_val[day] / CIR 

    return avg_S_val, avg_E_val, avg_I_val, avg_R_val, e 

def calculate_loss(BETA, S0, E0, I0, R0, CIR0, ft_df, tst_data, cnf_avg):
    _, _, _, _, e = generate_time_series(BETA, S0, E0, I0, R0, CIR0, ft_df, tst_data, 42, 70000000, 1 / 5.8, 1 / 5, 0.66)
    ALPHA = 1 / 5.8 

    e = ALPHA * e 
    e_val_avg = np.zeros(42)
    for i in range(42):
        cnt = 0
        for j in range(i, i - 7, -1):
            if j >= 0:
                cnt += 1
                e_val_avg[i] += e[j]
            else:
                break 
        e_val_avg[i] /= cnt 
    
    loss_val = 0
    for i in range(42):
        loss_val += (math.log(cnf_avg[i]) - math.log(e_val_avg[i])) ** 2 
    loss_val /= 42
    return loss_val 

def SEIR_without_Immunity_Wanning_Plot(parameters_, nors_days = 180):
    S_, E_, I_, R_, e_ = generate_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, False)
    plt.plot(S_, label = 'S')
    plt.plot(E_, label = 'E')
    plt.plot(I_, label = 'I')
    plt.plot(R_, label = 'R')
    plt.legend()
    plt.title('Without immunity waning after 180 days')
    plt.xlabel('Days since 16 Mar 2021')
    plt.ylabel('Number of people')
    plt.savefig('SEIR-Without-Immunity-Waning.jpg')

def SEIR_Vs_Immunity_Wanning_Plot(parameters_, ft_df, tst_data, nors_days = 180):
    S_, E_, I_, R_, e_ = generate_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True)
    plt.plot(S_, label = 'S')
    plt.plot(E_, label = 'E')
    plt.plot(I_, label = 'I')
    plt.plot(R_, label = 'R')
    plt.legend()
    plt.title('With immunity waning after 180 days')
    plt.xlabel('Days since 16 Mar 2021')
    plt.ylabel('Number of people')
    plt.savefig('SEIR-With-Immunity-Waning.jpg')

def future_time_series(B, S0, E0, I0, R0, CIR0, V, tst_data, nors_days, N, ALPHA, GAMMA, EPS, waning = True, closed_loop = False):
    S = np.zeros(nors_days)
    E = np.zeros(nors_days)
    I = np.zeros(nors_days)
    R = np.zeros(nors_days)
    e = np.zeros(nors_days)

    S[0] = S0 
    E[0] = E0
    I[0] = I0 
    R[0] = R0
    BETA = B
    everyday_new_cases = []
    for day in range(nors_days - 1):
        if closed_loop == True:
            if day % 7 == 1 and day >= 7:
                lst_week_avg_cases = 0
                for i in range(7):
                    CIR = CIR0 * tst_data[0][1] / tst_data[day - i][1] 
                    lst_week_avg_cases += ALPHA * (E[day - i]) / CIR 
                lst_week_avg_cases /= 7
                # adjust BETA
                if lst_week_avg_cases < 10000:
                    BETA = B 
                elif lst_week_avg_cases < 25000:
                    BETA = B * 2 / 3
                elif lst_week_avg_cases < 100000:
                    BETA = B / 2 
                else:
                    BETA = B / 3

        if day <= 30:
            delta_W = R0 / 30
        elif day >= 180:
            if waning == True:
                delta_W = R[day - 180] + EPS * V[day - 180][1]
            else:
                delta_W = 0
            pass
        else:
            delta_W = 0

        S[day + 1] = S[day] - BETA * S[day] * I[day] / N - EPS * V[day][1] + delta_W
        E[day + 1] = E[day] + BETA * S[day] * I[day] / N - ALPHA * E[day]
        I[day + 1] = I[day] + ALPHA * E[day] - GAMMA * I[day]
        R[day + 1] = R[day] + GAMMA * I[day] + EPS * V[day][1] - delta_W

        CIR = CIR0 * tst_data[0][1] / tst_data[day][1] 
        everyday_new_cases.append(ALPHA * E[day])
    
    avg_S_val = np.zeros(nors_days)
    avg_E_val = np.zeros(nors_days)
    avg_I_val = np.zeros(nors_days)
    avg_R_val = np.zeros(nors_days)

    for day in range(nors_days):
        cnt_avg_val = 0
        for prev_day in range(day, day - 7, -1):
            if prev_day >= 0:
                avg_S_val[day] += S[prev_day]
                avg_E_val[day] += E[prev_day]
                avg_I_val[day] += I[prev_day]
                avg_R_val[day] += R[prev_day]
                cnt_avg_val += 1
        avg_S_val[day] = avg_S_val[day] / cnt_avg_val
        avg_E_val[day] = avg_E_val[day] / cnt_avg_val
        avg_I_val[day] = avg_I_val[day] / cnt_avg_val
        avg_R_val[day] = avg_R_val[day] / cnt_avg_val

    for day in range(nors_days):
        CIR = CIR0 * tst_data[0][1] / tst_data[day][1] 
        e[day] = avg_E_val[day] / CIR 

    return avg_S_val, avg_E_val, avg_I_val, avg_R_val, e, everyday_new_cases

def newCases_OpenClosed_Plot(parameters_, ft_df, tst_data, grnd_truth, nors_days = 188):
    nors_days = 188
    plt.figure(figsize = (9, 7))

    _, _, _, _, _, everyday_new_cases = future_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(everyday_new_cases, label = 'ß, Open Loop')

    _, _, _, _, _, everyday_new_cases = future_time_series(parameters_[0] * 2 / 3, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(everyday_new_cases, label = '2/3 ß, Open Loop')

    _, _, _, _, _, everyday_new_cases = future_time_series(parameters_[0] / 2, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(everyday_new_cases, label = '1/2 ß, Open Loop')

    _, _, _, _, _, everyday_new_cases = future_time_series(parameters_[0] / 3, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(everyday_new_cases, label = '1/3 ß, Open Loop')

    _, _, _, _, _, everyday_new_cases = future_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=True)
    plt.plot(everyday_new_cases, label = 'Closed Loop')

    plt.plot(grnd_truth, label = 'Ground Truth (Reported Cases Only)')

    plt.legend()
    plt.title('Open & Closed Loop Predictions till 20th Sep 2021')
    plt.xlabel('Days since 16 Mar 2021')
    plt.ylabel('#new cases each day')
    plt.savefig('NewCases_EachDay_OpenClosed_Loop.jpg')
    print('Plot Save: \"NewCases_EachDay_OpenClosed_Loop.jpg\"')

def susceptible_OpenClosed_Plot(parameters_, ft_df, tst_data, nors_days = 188):
    nors_days = 188
    plt.figure(figsize = (9, 7))

    S_, _, _, _, _, _ = future_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(S_, label = 'ß, Open Loop')

    S_, _, _, _, _, _ = future_time_series(parameters_[0] * 2 / 3, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(S_, label = '2/3 ß, Open Loop')

    S_, _, _, _, _, _ = future_time_series(parameters_[0] / 2, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(S_, label = '1/2 ß, Open Loop')

    S_, _, _, _, _, _ = future_time_series(parameters_[0] / 3, parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=False)
    plt.plot(S_, label = '1/3 ß, Open Loop')

    S_, _, _, _, _, _ = future_time_series(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data, nors_days, 70000000, 1/5.8, 1/5, 0.66, True, closed_loop=True)
    plt.plot(S_, label = 'Closed Loop')

    plt.legend()
    plt.title('Open & Closed Loop Predictions till 20th Sep 2021')
    plt.xlabel('Days since 16 Mar 2021')
    plt.ylabel('#susceptible peoples')
    plt.savefig('Susceptible_OpenClosed_Loop.jpg')
    print('Plot Save: \"Susceptible_OpenClosed_Loop.jpg\"')

if __name__ == "__main__":
    
    data, full_df = preprocessing()
    st_date = datetime.datetime(2021, 3, 16)
    end_date = datetime.datetime(2021, 4, 26)
    lt_date = datetime.datetime(2021, 12, 31)
    
    ft_df, tst_data_df, cnf_avg, grd_truth = confirmedAvgCases(data, full_df, lt_date)
    
    parameters_ = np.array([4.49722551e-01, 4.89999999e+07, 7.69999180e+04, 7.69999182e+04, 2.08529999e+07, 1.28716990e+01])

    loss = calculate_loss(parameters_[0], parameters_[1], parameters_[2], parameters_[3], parameters_[4], parameters_[5], ft_df, tst_data_df, cnf_avg)
    print(f'Loss Value = {loss}')
    
    SEIR_Vs_Immunity_Wanning_Plot(parameters_, ft_df, tst_data_df)

    newCases_OpenClosed_Plot(parameters_, ft_df, tst_data_df, grd_truth)
    susceptible_OpenClosed_Plot(parameters_, ft_df, tst_data_df, grd_truth)