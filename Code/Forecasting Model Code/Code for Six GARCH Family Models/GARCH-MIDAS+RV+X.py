# -*- coding: utf-8 -*-

from numpy import array,nan, inf, nanmean, nansum, zeros, ones, arange, \
    log, pi, vstack, abs, diag, isnan, isinf
from numpy.linalg import pinv
from pandas import DataFrame
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
from statsmodels.tools.numdiff import approx_hess
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # 忽略警告


def Garch_Midas_RV_X(data, indicator, period1, period2):
    mu0 = np.mean(data['rt'])
    alpha0 = 0.1
    beta0 = 0.9
    theta1 = 0.1
    theta2 = 0.1
    omega = 0.1
    m0 = 0.01
    w1 = 3
    w2 = 5

    n = len(data)
    rt_full = data['rt']
    sample = int(n * 0.7)
    rt_sample = rt_full[:sample]
    nobs = len(rt_sample)

    months = 1 + int((data['year'][n - 1] - data['year'][0]) * 12 + (data['month'][n - 1] - data['month'][0]) % 12)
    rvt = zeros(months)
    year = data['year'][0]
    month = data['month'][0]
    for t in range(months):
        rvt[t] = sum(data.loc[(data['year'] == year) & (data['month'] == month)]['rt'] ** 2)##
        month = month + 1
        if month == 13:
            month = 1
            year = year + 1

    year = data['year'][0] + int(period1 / 12)
    month = data['month'][0] + period1 % 12
    if month > 12:
        month = month - 12
        year = year + 1
    loop_start1 = np.where((data['year'] == year) & (data['month'] == month))[0][0]
    for t in range(period1, months):
        data.loc[(data['year'] == year) & (data['month'] == month), 'rv'] = t
        month = month + 1
        if month == 13:
            month = 1
            year = year + 1

    year = data['year'][0] + int(period2 / 12)
    month = data['month'][0] + period2 % 12
    if month > 12:
        month = month - 12
        year = year + 1

    loop_start2 = np.where((data['year'] == year) & (data['month'] == month))[0][0]
    for t in range(period2, months):
        data.loc[(data['year'] == year) & (data['month'] == month), 'indicator'] = t
        month = month + 1
        if month == 13:
            month = 1
            year = year + 1

    rv = data['rv']
    rv_sample = rv[:sample]
    indicator_i = data['indicator']
    indic_sample = indicator_i[:sample]
    loop_start = max(loop_start1, loop_start2)

    params0 = [alpha0, beta0, theta1, theta2, omega, m0, mu0, w1, w2]
    bound = [(0, 0.1), (0.8, 1), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (1, inf), (1, inf)]
    options = {'maxiter': 3000, 'disp': True}
    myfun = lambda x: -nansum(f_ml(x, data, indic_sample, rt_sample, rv_sample, rvt, indicator,
                                   period1, period2, nobs)[0])
    cons = {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}
    est_params = minimize(myfun, params0, method='SLSQP', options=options, bounds=bound, constraints=cons).x

    """检验"""
    hess = approx_hess(est_params, myfun)
    vcv_h = pinv(hess)
    se_h = abs(diag(vcv_h)) ** 0.5
    z_stat = est_params / se_h
    p_value = st.t.sf(abs(z_stat), n - len(est_params))
    p_value[p_value < 1e-6] = 0

    index = ['alpha', 'beta', 'theta1', 'theta2', 'omega', 'm', 'mu', 'w1', 'w2']
    columns = ['result', 'stderr', ' stat', 'p_value']
    result = vstack((est_params, se_h, z_stat, p_value)).T
    table = DataFrame(result, index=index, columns=columns)
    print(table)

    aic = (18 - nansum(f_ml(est_params, data, indic_sample, rt_sample, rv_sample, rvt, indicator,
                            period1, period2, nobs)[0]) * 2) / nobs
    bic = (-2 * nansum(f_ml(est_params, data, indic_sample, rt_sample, rv_sample, rvt, indicator,
                            period1, period2, nobs)[0]) + 9 * np.log(nobs)) / nobs
    print('\nAIC : %f\n' % aic)
    print('BIC : %f\n' % bic)

    variance, long_run, short_run = f_ml(est_params, data, indicator_i, rt_full, rv, rvt, indicator,
                                         period1, period2, n)[1:]
    realized_y2 = rt_full ** 2
    forecast_error = variance - realized_y2
    est_sample_mse = nanmean(forecast_error[loop_start:sample] ** 2)
    print('MSE of one-step variance forecast (period 1 to %d): %f \n' % (sample, est_sample_mse))
    if sample < n:
        out_sample_mse = nanmean(forecast_error[sample:] ** 2)
        print('MSE of one-step variance forecast (period %d to %d):%f \n' % (sample + 1, n, out_sample_mse))
    return est_params, variance, long_run, short_run

def weights(k, param1):     # k为低频变量的最大滞后阶数
    eps = 1e-16  # 最小的数
    seq = arange(k, 0, -1)
    weight = (1 - seq / k + 10 * eps) ** (param1 - 1)
    weight = weight / nansum(weight)
    return weight


def f_ml(params, data, indic, rt, rv, rvt, indicator, period1, period2, nobs):
    alpha0 = params[0]
    beta0 = params[1]
    theta1 = params[2]
    theta2 = params[3]
    omega0 = params[4]
    m0 = params[5]
    mu0 = params[6]
    w1 = params[7]
    w2 = params[8]

    weight1 = weights(period1, w1)
    weight2 = weights(period2, w2)
    res = (rt - mu0) ** 2
    short_run = ones(nobs)
    tau = zeros(nobs)
    tau_avg = m0 + theta1 * nanmean(rvt) + theta2 * nanmean(indicator)
    variance = tau_avg * ones(nobs)

    period = max(period1, period2)
    year = data['year'][0] + int(period / 12)
    month = data['month'][0] + period % 12
    if month > 12:
        month = month - 12
        year = year + 1
    loop_start = np.where((data['year'] == year) & (data['month'] == month))[0][0]
    for t in range(loop_start, nobs):
        num1 = int(rv[t])
        num2 = int(indic[t])
        tau[t] = m0 + theta1 * weight1.dot(rvt[num1-period1:num1]) + theta2 * weight2.dot(indicator[num2-period2:num2])
        short_run[t] = omega0 + alpha0 * res[t] / tau[t] + beta0 * short_run[t - 1]
        variance[t] = tau[t] * short_run[t]
    if any(variance < 0):
        logL = array([-1e10] * nobs)
        Variance = array([nan] * nobs)
        ShortRun = array([nan] * nobs)
        LongRun = array([nan, nobs])
        return logL, Variance, ShortRun, LongRun
    log_l = -0.5 * (log(2 * pi * variance + 1e-3) + res / variance)
    log_l[:loop_start] = 0
    if isnan(log_l.sum()) or isinf(log_l.sum()):
        log_l = array([-1e10] * nobs)
    long_run = tau
    return log_l, variance, long_run, short_run



if __name__ == '__main__':
    data_test = pd.read_csv(r'DT50低频.csv',encoding="gbk")
    data2 = pd.read_csv(r'EPU.csv',encoding="gbk")
    GEPU_monthly = data2['EPU']#经济政策的不确定性指数
    est_params,variance,long_run, short_run=Garch_Midas_RV_X(data_test, GEPU_monthly, 12, 12)

####
n = len(data_test)
date =data_test["date"].values
rt = data_test['rt'].values
sample = int(n * 0.7)

date=date[sample:]
yf = rt[sample:]
yf = yf ** 2
Variance_f = variance[sample:]

d=[yf,Variance_f]
df = pd.DataFrame({'v':yf, 'GARCH-MIDAS+RV+X':Variance_f},index=date)
df.to_csv('C:/Users/赵晨晨/Desktop/DT50样本外/7.csv')

import matplotlib.pylab as plt
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title('nihe')

plt.subplot(312)
plt.plot(variance, 'b', long_run, 'r')
plt.subplot(313)
plt.plot(variance, 'b', short_run, 'y')
plt.show()
