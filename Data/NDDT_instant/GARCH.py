# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:42:53 2019

@author: 刘国山
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import warnings
warnings.filterwarnings("ignore") #忽略警告
import numpy as np
import pandas as pd
from numpy.linalg import inv,pinv
from scipy.stats import norm,t
from scipy.optimize import minimize
from numpy import inf,array,nan,nansum
from statsmodels.tools.numdiff import approx_hess

model_name='garch'
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

time_line = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))

# print(os.getcwd())
log_path=os.path.join(os.getcwd(),f'output/log/{model_name}')
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

def garch(rt,estSample):
    mu = np.mean(rt)
    alpha = 0.3
    beta = 0.8
    omega = 0.02
    rt = rt.flatten()
    rtFull = rt[:]
    rt = rt[:estSample]
    nobs = len(rt)
    params0 = [mu,omega,alpha,beta]
    f = lambda x:-nansum(fML(x,rt,nobs)[0])
    bound = [(-inf,inf),(-inf,inf),(0,1),(0,1)]
    cons = {'type':'ineq','fun':lambda x:1-x[2]-x[3]}
    estParams = minimize(f,params0,method='SLSQP',bounds=bound,constraints=cons).x
    
    hess = approx_hess(estParams,f)
    #vcv_H = np.linalg.inv(hess)
    vcv_H = pinv(hess)
    se_H = abs(np.diag(vcv_H))**0.5
    zstat = estParams/se_H
    p_value = t.sf(abs(zstat),len(rt)-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','omega','alpha','beta']
    columns = ['result','stderr','stat','p_value']
    result = np.vstack((estParams,se_H,zstat,p_value)).T
    table = pd.DataFrame(result,index=index,columns=columns)
    logger.info(table)
    
    AIC = (8-nansum(fML(estParams,rt,nobs)[0])*2)/len(rt)
    BIC = (-2*nansum(fML(estParams,rt,nobs)[0])+4*np.log(len(rt)))/len(rt)
    logger.info('AIC:\t%f\n'%(AIC))
    logger.info('HQ:\t%f\n'%(BIC))
    
    nobsBig = len(rtFull)
    Variance = fML(estParams,rtFull,nobsBig)[1]
    realizedY2 = rtFull**2
    forecasterror = Variance - realizedY2
    estSampleMse = np.nanmean(forecasterror[:estSample]**2)
    logger.info('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = np.nanmean(forecasterror[estSample:] ** 2)
        logger.info('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance

def fML(params,rt,nobs):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    epsilon = rt-mu
    intercept = omega
    if alpha<0 or alpha>1 or beta<0 or beta>1:
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        return logL,Variance
    Variance = np.ones(len(rt))#*(omega/(1-alpha-beta))
    for i in range(1,len(rt)):
        Variance[i] = intercept + alpha*epsilon[i-1]**2 + beta*Variance[i-1] 
    if any(Variance<0):
        logL = np.array([-1e10]*len(rt))
        Variance = np.array([np.nan]*len(rt))
        return logL,Variance
    #L = norm.pdf(y,loc=mu,scale=np.sqrt(v))
    #logL = np.sum(np.log(L))
    logL = -0.5*(np.log(2*np.pi*Variance+1e-3)+(rt-mu)**2/(Variance))
    if np.isnan(logL.sum()) or np.isinf(logL.sum()):
        logL = np.array([-1e10] * nobs)
    return logL,Variance
if __name__=='__main__':
    data = pd.read_csv(r'nddt_lowf.csv',encoding="gbk")
    rt = data["rt"].values
    insample = 1569
    params, Variance = garch(rt, insample)#1075是测试样本



import matplotlib.pyplot as plt
yf = rt[insample:]
yf = yf ** 2
Variance_f = Variance[insample:]
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title('garch')

plt.savefig(f'output/pic/{model_name}.png')
plt.show()
d = [yf, Variance_f]
df = pd.DataFrame({'v': yf, 'GARCH': Variance_f})
df.to_csv(f'output/csv/{model_name}.csv')