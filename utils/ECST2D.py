#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 18:26:45 2021

@author: martin
"""

import numpy as np
import mat73
from numpy.linalg import inv
from scipy.fftpack import dct
from sklearn.linear_model import Lasso
import spgl1
from scipy.stats import bernoulli


def cstv1(M,y,mtxPsi='haar',alea=True,tiporand='G',valpha=1e-5,fNtr=1):
    '''
    CSTv1 Esta función intenta resolver el sistema de ecuaciones M*x=y
    usando el enfoque de "compressed sensing". Como resolvedor usa la 
    minimización L1 LASSO del paquete python scikit-learn. 
    # Ver: https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html
    '''
    # Extrayendo datos de los argumentos
    Ns = np.size(y,0) # cantidad de sensores
    Nt = np.size(y,1) # cantidad de muestras 
    N = np.size(M,1) # cantidad de pixeles
        
 
    if alea:       
        if tiporand == 'G': # Aplicación Matriz Aleatoria Gaussiana
            Ntr = int(Nt*fNtr)
            sampleTimes = np.arange(0,Nt,dtype=int);
            Mr=np.zeros((Ns*Ntr,N));
            yr=np.zeros((Ns,Ntr))
            for k in range(1,Ns+1):
                idx = np.random.permutation(sampleTimes);
                sT = idx[0:Ntr]
                aux1=Nt*(k-1)
                aux2=Ntr*(k-1)
                Maux=M[aux1:aux1+Nt,0:];
                Mr[aux2:aux2+Ntr,0:]=Maux[sT,0:];
                yr[k-1,:] = y[k-1,sT];
            y = yr
            M = Mr
            Nt = Ntr
            del Maux,Mr,yr,aux1,sampleTimes,idx
                
        else: # Aplicación Matriz Aleatoria de Bernoulli
            Ntr = int(Nt*fNtr)
            sampleTimes = np.arange(0,Nt,dtype=int);
            Mr=np.zeros((Ns*Ntr,N));
            yr=np.zeros((Ns,Ntr))
            for k in range(1,Ns+1):
                bv = bernoulli.rvs(fNtr*1.05,size=Nt)
                idx = (sampleTimes+1)*bv
                sT = idx[0:Ntr]
                aux1=Nt*(k-1)
                aux2=Ntr*(k-1)
                Maux=M[aux1:aux1+Nt,0:];
                Mr[aux2:aux2+Ntr,0:]=Maux[sT,0:];
                yr[k-1,:] = y[k-1,sT];
            y = yr
            M = Mr
            Nt = Ntr
            del Maux,Mr,yr,aux1,sampleTimes,idx
    
    # Ensamblado matriz de transformación
    if mtxPsi == 'haar':
        MPsi=haarmtx(N); 
        iMPsi=np.transpose(MPsi)
    elif mtxPsi == 'DCT':
        MPsi = dct(np.eye(N), norm='ortho', axis=0)
        iMPsi = inv(MPsi)
    elif mtxPsi == 'diff64':
        WMTXdic = mat73.loadmat('MatricesWavelets/mtx64diff.mat')
        MPsi = WMTXdic['diff64']
        iMPsi = WMTXdic['idiff64']    
    elif mtxPsi == 'db464':
        WMTXdic = mat73.loadmat('MatricesWavelets/mtx64db4.mat')
        MPsi = WMTXdic['db464']
        iMPsi = WMTXdic['idb464']
    elif mtxPsi == 'sym264':
        WMTXdic = mat73.loadmat('MatricesWavelets/mtx64sym2.mat')
        MPsi = WMTXdic['sym264']
        iMPsi = WMTXdic['isym264']
    else:
        MPsi = np.eye(N)
        iMPsi = MPsi
    
    # resolvedor usa la minimización L1 LASSO del paquete python scikit-learn. 
    # Ver: https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html
    y = np.reshape(y,(Nt*Ns,));
    MW = M@MPsi
    
    #valpha = 1e-6;   # Desired ||x||_1
    lasso=Lasso(alpha=valpha)
    lasso.fit(MW,y)
    theta=np.array(lasso.coef_)
    x = MPsi@theta 
       
    return(x)
