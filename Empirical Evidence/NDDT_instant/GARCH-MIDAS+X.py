# -*- coding: utf-8 -*-
# coding:unicode_escape
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
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

model_name='garch_midas_x'
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

time_line = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))

# print(os.getcwd())
log_path=os.path.join(os.getcwd(),'output/log/'+model_name)
if not os.path.exists(log_path):
    os.makedirs(log_path)
logfile = log_path + '/'+time_line + '.txt'

handler = logging.FileHandler(logfile,mode='w') # 输出到log文件的handler
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s\n")
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s:\n %(message)s\n")
handler.setFormatter(formatter)

console = logging.StreamHandler() # 输出到console的handler
console.setLevel(logging.WARNING)

logger.addHandler(handler)
logger.addHandler(console)

def Garch_Midas_X(data, indicator, period):
    mu0 = np.mean(data['rt'])
    #alpha0 = 0.1
    alpha0 = 0.1
    beta0 = 0.9
    theta0 = 0.1
    omega = 0.1
    m0 = 0.1
    w1 = 3

    n = len(data)
    rt_full = data['rt']
    sample = int(n * 0.7)
    rt_sample = rt_full[:sample]
    nobs = len(rt_sample)

    months = 1 + int((data['year'][n - 1] - data['year'][0]) * 12 + (data['month'][n - 1] - data['month'][0]) % 12)

    year = data['year'][0] + int(period / 12)
    month = data['month'][0] + period % 12
    if month > 12:
        month = month - 12
        year = year + 1

    loop_start = np.where((data['year'] == year) & (data['month'] == month))[0][0]
    for t in range(period, months):
        data.loc[(data['year'] == year) & (data['month'] == month), 'indicator'] = t
        month = month + 1
        if month == 13:
            month = 1
            year = year + 1

    indicator_i = data['indicator']
    indic_sample = indicator_i[:sample]

    params0 = [alpha0, beta0, theta0, omega, m0, mu0, w1]
    bound = [(0, 0.1), (0.8, 1), (-5, 5), (-5, 5), (-5, 5), (-5, 5), (1, inf)]
    #bound = [(0, 1), (0, 1),(-5, 5), (0, 1),  (-5, 5), (-5, 5), (1, inf)]
    options = {'maxiter': 3000, 'disp': True}
    myfun = lambda x: -nansum(f_ml(x, data, indic_sample, rt_sample, indicator, period, nobs)[0])
    cons = {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}
    est_params = minimize(myfun, params0, method='SLSQP', options=options, bounds=bound, constraints=cons).x

    """检验"""
    hess = approx_hess(est_params, myfun)
    vcv_h = pinv(hess)
    se_h = abs(diag(vcv_h)) ** 0.5
    z_stat = est_params / se_h
    p_value = st.t.sf(abs(z_stat), nobs - len(est_params))
    p_value[p_value < 1e-6] = 0

    index = ['alpha', 'beta', 'theta', 'omega', 'm', 'mu', 'w1']
    columns = ['result', 'stderr', ' stat', 'p_value']
    result = vstack((est_params, se_h, z_stat, p_value)).T
    table = DataFrame(result, index=index, columns=columns)
    logger.info(table)

    aic = (14 - nansum(f_ml(est_params, data, indic_sample, rt_sample, indicator, period, nobs)[0]) * 2) / nobs
    bic = (-2 * nansum(f_ml(est_params, data, indic_sample, rt_sample, indicator, period, nobs)[0])
           + 7 * np.log(nobs)) / nobs
    logger.info('\nAIC : %f\n' % aic)
    logger.info('BIC : %f\n' % bic)

    variance, long_run, short_run = f_ml(est_params, data, indicator_i, rt_full, indicator, period, n)[1:]
    realized_y2 = rt_full ** 2
    forecast_error = variance - realized_y2
    est_sample_mse = nanmean(forecast_error[loop_start:sample] ** 2)
    logger.info('MSE of one-step variance forecast (period 1 to %d): %f \n' % (sample, est_sample_mse))
    if sample < n:
        out_sample_mse = nanmean(forecast_error[sample:] ** 2)
        logger.info('MSE of one-step variance forecast (period %d to %d):%f \n' % (sample + 1, n, out_sample_mse))
    return est_params, variance, long_run, short_run

def weights(k, param1):     # k为低频变量的最大滞后阶数
    eps = 1e-16  # 最小的数
    seq = arange(k, 0, -1)
    weight = (1 - seq / k + 10 * eps) ** (param1 - 1)
    weight = weight / nansum(weight)
    return weight


def f_ml(params, data, indic, rt, indicator, period, nobs):
    alpha0 = params[0]
    beta0 = params[1]
    theta0 = params[2]
    omega0 = params[3]
    m0 = params[4]
    mu0 = params[5]
    w1 = params[6]

    weight = weights(period, w1)
    res = (rt - mu0) ** 2
    short_run = ones(nobs)
    tau = zeros(nobs)
    tau_avg = m0 + theta0 * nanmean(indicator)
    variance = tau_avg * ones(nobs)

    year = data['year'][0] + int(period / 12)
    month = data['month'][0] + period % 12
    if month > 12:
        month = month - 12
        year = year + 1
    loop_start = np.where((data['year'] == year) & (data['month'] == month))[0][0]
    for t in range(loop_start, nobs):
        num = int(indic[t])
        tau[t] = m0 + theta0 * weight.dot(indicator[num-period:num])
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
    data_test = pd.read_csv(r'nddt_lowf.csv')
    data2 = pd.read_csv(r's_index.csv',encoding="gbk")
    GEPU_monthly = data2['sentiment_index']
    insample = 1883
    est_params,variance, long_run, short_run =Garch_Midas_X(data_test, GEPU_monthly, 12)

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
df = pd.DataFrame({'v':yf, 'GARCH-MIDAS+X':Variance_f},index=date)
df.to_csv(f'output/csv/{model_name}.csv')

import matplotlib.pylab as plt
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title(model_name)

plt.subplot(312)
plt.plot(variance, 'b', long_run, 'r')
plt.subplot(313)
plt.plot(variance, 'b', short_run, 'y')
plt.savefig(f'output/pic/{model_name}.png')
plt.show()
