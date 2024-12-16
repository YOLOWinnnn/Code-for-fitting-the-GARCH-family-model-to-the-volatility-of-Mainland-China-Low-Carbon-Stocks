# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:36:46 2019

@author: 刘国山
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.stats import norm,t
from scipy.optimize import minimize
from numpy import inf,log,pi,nansum,exp
from statsmodels.tools.numdiff import approx_hess
import warnings
warnings.filterwarnings("ignore") #忽略警告
model_name='rgarch'
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

def Rgarch(rt,rk,estSample):
    beta = 0.8
    alpha = 0.05
    omega = 0.02
    #omega = 0.01
    xi = -0.2
    psi = 0.01
    tao1 = -0.004
    tao2 = 0.02
    deta_u = 2.5
    mu = np.mean(rt)
    rtFull = rt[:]
    rt = rt[:estSample]
    rkFull = rk[:]
    rk = rk[:estSample]
    nobs = len(rt)
    nobsBig = len(rtFull)
    params0 = [mu,omega,alpha,beta,xi,psi,tao1,tao2,deta_u]
    f = lambda x:-nansum(fML(x,rt,rk,nobs)[0])
    bound = [(-inf,inf),(-inf,inf),(0,0.4),(0.7,1),(-inf,inf),(0,1),(-inf,0),(0,1),(0,3)]
    options = {'maxiter':3000,'disp':True}
    cons = {'type':'ineq','fun':lambda x:1-x[2]-x[3]}
    estParams = minimize(f,params0,method='SLSQP',options=options,bounds=bound,constraints=cons).x
    
    hess = approx_hess(estParams,f)
    vcv_H = pinv(hess)
    se_H = abs(np.diag(vcv_H))**0.5
    zstat = estParams/se_H
    p_value = t.sf(abs(zstat),len(rt)-len(estParams))
    p_value[p_value<1e-4]=0
    
    index = ['mu','omega','alpha','beta','xi','psi','tao1','tao2','deta_u']
    columns = ['result','stderr','stat','p_value']
    result = np.vstack((estParams,se_H,zstat,p_value)).T
    table = pd.DataFrame(result,index=index,columns=columns)
    logger.info(table)

    AIC = (18-nansum(fML(estParams,rt,rk,nobs)[0])*2)/len(rt)
    BIC = (-2*nansum(fML(estParams,rt,rk,nobs)[0])+9*np.log(len(rt)))/len(rt)
    logger.info('AIC:\t%f\n'%(AIC))
    logger.info('HQ:\t%f\n'%(BIC))
    

    Variance = fML(estParams,rtFull,rkFull,nobsBig)[1]
    realizedY2 = rtFull**2
    forecasterror = Variance - realizedY2
    estSampleMse = np.nanmean(forecasterror[:estSample]**2)
    logger.info('MSE of one-step variance forecast (period 1 to %d):%f \n'%(estSample,estSampleMse))
    if estSample<nobsBig:
        outSampleMse = np.nanmean(forecasterror[estSample:] ** 2)
        logger.info('MSE of one-step variance forecast (period %d to %d):%f \n'%(estSample+1,nobsBig,outSampleMse))
    return estParams,Variance

def fML(params,rt,rk,nobs):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    xi = params[4]
    psi = params[5]
    tao1 = params[6]
    tao2 = params[7]
    deta_u = params[8]
    intercept = omega
    ht = np.ones(len(rt))
    Zt = np.ones(len(rt))
    ut = np.ones(len(rt))
    #Variance = np.ones(len(rt))
    for i in range(1,len(rt)):
        ht[i] = intercept + beta*ht[i-1] + alpha*rk[i-1]
        Zt[i] = (rt[i]-mu)/np.sqrt(ht[i])
        ut[i] = rk[i] - xi - psi*ht[i] - tao1*Zt[i] - tao2*(Zt[i]**2) + tao2
    Variance = ht
    if any(Variance<0):
        logL = np.array([-1e10]*len(rt))
        Variance = np.array([np.nan]*len(rt))
        return logL,Variance
    #L1 = norm.pdf(Zt,loc=0,scale=1)
    #L2 = norm.pdf(ut,loc=0,scale=deta_u)
    #logL = np.log(L1*L2)
    logL = -0.5*(log(2*pi*ht+1e-3)+Zt**2)-0.5*(log(2*pi*deta_u**2+1e-3)+ut**2/deta_u**2)
    if np.isnan(logL.sum()) or np.isinf(logL.sum()):
        logL = np.array([-1e10]*nobs)
    return logL,Variance
if __name__=='__main__':
    data = pd.read_csv(r'nddt_lowf.csv',encoding="gbk")
    #rt = data.iloc[:,2].values
    #rk = data.iloc[:,3].values
    rt = data["rt"].values
    rk = data["rk"].values
    insample = 1569
    params,Variance = Rgarch(rt,rk,insample)#共5487



import matplotlib.pyplot as plt
yf = rt[insample:]
yf = yf ** 2
Variance_f = Variance[insample:]
plt.subplot(311)
plt.plot(yf, 'b', Variance_f, 'r')
plt.title(model_name)
plt.savefig(f'output/pic/{model_name}.png')
plt.show()

d = [yf, Variance_f]
df = pd.DataFrame({'v': yf, 'RGARCH': Variance_f})
df.to_csv(f'output/csv/{model_name}.csv')