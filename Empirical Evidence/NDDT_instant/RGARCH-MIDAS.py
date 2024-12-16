# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:32:23 2019

@author: 刘国山
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from numpy import array,nan,inf,nanmean,nansum,ceil,zeros,ones,exp,arange,any,log,pi,vstack,\
    all,max,abs,diag,isnan,isinf
from numpy.linalg import inv,pinv
from pandas import DataFrame
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as st
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess
import numpy as np
import warnings
warnings.filterwarnings("ignore") #忽略警告

model_name='rgarch_midas'
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

def RGarchMidas(rt,rk,K,period,estSample):
    mu0 = np.nanmean(rt);beta0 = 1;alpha0 = 0.1;
    xi = -0.2
    #psi = 0.01
    psi = 0.15
    tao1 = 0.1
    #tao2 = 0.02
    tao2 = 0.1
    deta_u = 1
    #theta0 = 0.01
    theta0 = 0.1
    omega = 0.06;m0 = 0.1;w1=2.5
    rtFull = rt[:]
    rt = rt[:estSample]
    rkFull = rk[:]
    rk = rk[:estSample]
    nobs = len(rt)
    RV = zeros(nobs)
    for t in range(period,nobs):
        RV[t] = nansum(rt[t-period:t]**2)
    RV[:period] = RV[period]
    params0 = [mu0,omega,alpha0,beta0,xi,psi,tao1,tao2,m0,theta0,deta_u,w1]
    bound = [(-inf,inf),(-inf,inf),(0,0.2),(0.7,1),(-inf,inf),(0,1),(-inf,0),(0,1),(0,2),(-inf,inf),(0,inf),(1.001,50)]
    options = {'maxiter':3000,'disp':True}
    cons = {'type':'ineq','fun':lambda x:1-x[2]-x[3]}
    myfun = lambda x:-nansum(fML(x,rt,rk,RV,K,period,nobs)[0])
    estParams = minimize(myfun,params0,method='SLSQP',options=options,bounds=bound,constraints=cons).x
    """检验"""
    hess = approx_hess(estParams,myfun)
    vcv_H = pinv(hess)
    se_H = abs(diag(vcv_H))**0.5
    zstat = estParams/se_H
    p_value = st.t.sf(abs(zstat),nobs-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','omega','alpha','beta','xi','psi','tao1','tao2','m','theta','deta_u','w1']
    columns = ['result','stderr','stat','p_value']
    result = vstack((estParams,se_H,zstat,p_value)).T
    table = DataFrame(result,index=index,columns=columns)
    logger.info(table)
	
    AIC = (24-nansum(fML(estParams,rt,rk,RV,K,period,nobs)[0])*2)/len(rt)
    BIC = (-2*nansum(fML(estParams,rt,rk,RV,K,period,nobs)[0])+12*np.log(len(rt)))/len(rt)
    logger.info('AIC是：%f\n'%(AIC))
    logger.info('HQ是：%f\n'%(BIC))
    nobsBig = len(rtFull)
    RVBig = zeros(nobsBig)
    for t in range(period,nobsBig):
        RVBig[t] = nansum(rtFull[t-period:t]**2)
    RVBig[:period] = RVBig[period]
    Variance,LongRun,ShortRun = fML(estParams,rtFull,rkFull,RVBig,K,period,nobsBig)[1:]
    realizedrt2 = rtFull**2
    forecasterror = Variance - realizedrt2
    estSampleMse = nanmean(forecasterror[:estSample]**2)
    logger.info('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = nanmean(forecasterror[estSample:] ** 2)
        logger.info('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance,LongRun,ShortRun

	
def BetaWeights(K,param1):
    #eps = 2.2204e-16#最小的数
    eps =1e-16
    seq = arange(K,0,-1)
    weights = (1-seq/K+10*eps)**(param1-1)
    weights = weights/nansum(weights)
    return weights

def fML(params,rt,rk,RV,K,period,nobs):
    mu0 = params[0]
    omega = params[1]
    alpha0 = params[2]
    beta0 = params[3]
    xi = params[4]
    psi = params[5]
    tao1 = params[6]
    tao2 = params[7]
    m0 = params[8]
    theta0 = params[9]
    deta_u = params[10]
    w0 = params[11]
    theta0 = theta0**2
    m0 = m0**2
    intercept = omega
    if alpha0<0 or alpha0>1 or beta0<0 or beta0>1:
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun
    ShortRun = ones(nobs)
    tauAvg = exp(m0+theta0*nanmean(RV))
    Variance = tauAvg*ones(nobs)
    zt = np.ones(nobs)
    ut = np.ones(nobs)
    nlagBig = period*K
    weights = BetaWeights(nlagBig,w0)
    loopStart = nlagBig
    for t in range(loopStart,nobs):
        tau = m0 +theta0*(weights.dot(RV[t-nlagBig:t]))
        ShortRun[t] = intercept+alpha0*rk[t-1]+beta0*ShortRun[t-1]
        Variance[t] = tau*ShortRun[t]
        zt[t] = (rt[t]-mu0)/np.sqrt(Variance[t])        
        ut[t] = rk[t]-xi-psi*Variance[t]- tao1*zt[t] - tao2*(zt[t]**2) + tao2
        
    if any(Variance<0):
        logL = array([-1e10]*nobs)
        Variance = array([nan]*nobs)
        ShortRun = array([nan]*nobs)
        LongRun = array([nan,nobs])
        return logL,Variance,ShortRun,LongRun
    """
    L1 = norm.pdf(zt,loc=0,scale=1)
    L2 = norm.pdf(ut,loc=0,scale=deta_u)
    logL = np.log(L1*L2)
    logL[:period*K] = 0
    logL=np.sum(logL)
    """
    logL = -0.5*(log(2*pi*Variance+1e-3)+zt**2)-0.5*(log(2*pi*deta_u**2+1e-3)+ut**2/deta_u**2)
    logL[:period*K] = 0
    if isnan(logL.sum()) or isinf(logL.sum()):
        logL = array([-1e10]*nobs)
    LongRun = Variance/ShortRun
    return logL,Variance,ShortRun,LongRun

if __name__=='__main__':
    data = pd.read_csv(r'nddt_lowf.csv',encoding="gbk")
    rt = data["rt"].values
    rk = data["rk"].values
    insample = 1569
    estParams,Variance, LongRun, ShortRun = RGarchMidas(rt,rk,12, 22,insample)

    import matplotlib.pyplot as plt
    yf = rt[insample:]
    yf = yf ** 2
    Variance_f = Variance[insample:]
    plt.subplot(311)
    plt.plot(yf, 'b', Variance_f, 'r')
    plt.title(model_name)

    plt.subplot(312)
    plt.plot(Variance, 'b', LongRun, 'r')
    plt.subplot(313)
    plt.plot(Variance, 'b', ShortRun, 'y')
    plt.savefig(f'output/pic/{model_name}.png')
    plt.show()

    d = [yf, Variance_f]
    df = pd.DataFrame({'v': yf, 'RGARCH-MIDAS': Variance_f})
    df.to_csv(f'output/csv/{model_name}.csv')