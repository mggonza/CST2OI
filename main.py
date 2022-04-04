#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 20:24:59 2021

@author: martin
"""

# Módulos o paquetes
import numpy as np
from utils.modeloT2D import mbt2DLSv5 
from utils.ECST2D import cstv1
from utils.BP2D import BPv1
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from scipy import stats
from phantominator import shepp_logan
from utils.fantom import phantomfun
import math


# Armado de la matriz modelo
Ns=5;  # cantidad de sensores 10
Nt=4096; # cantidad de muestras 2048
#nx=128; dx=50e-6; # tamaño pixel
nx=64; dx=50e-6;
#N=nx*nx; # pixeles totales
    
cr=10e-3; # Radio circunferencia de detección
vs=1500; # velocidad del sonido del medio
ls=0; #20e-3; # largo sensor lineal
nls=1;#300; # cantidad de elementos que componen el sensor lineal
arco=360; # arco de circunferencia donde se disponene equiespaciadamente los sensores
tim=dx*nx; # Tamaño de la imagen tim x tim

# inicialización vectores y matrices 
Pd = np.zeros((Ns,Nt)); # vector de mediciones

# Eje de tiempo
to=(cr-tim/2)/vs*0.5; tf=(cr+2*tim)/vs; # tiempo inicial y final
tf=tf/np.cos(np.arctan(ls/vs/tf)) # correccion cuando el sensor es lineal
tproc=np.linspace(to,tf,Nt);

# Angulos donde son dispuestos los sensores
# Distribución uniforme 
th = np.linspace(0,arco*np.pi/180,Ns+1); th = th[0:len(th)-1];

# Armado matriz modelo
Ao=mbt2DLSv5(nx,dx,vs,Ns,cr,arco,ls,nls,tproc,th); 

# Imagenes a probar    
imagen = ('TOA64','Derenzo64','bloodvessel64','breast64','SL')

# Matrices de transformación a probar
mtrans = ('Iden','diff64','haar','db464','sym264','DCT') 

# Aleatorizar muestras?
alea = ( False, True, True )

# Inicializacion matrices de méttricas
PC = np.zeros((len(alea),len(mtrans)+1,len(imagen)))
RMSE = np.zeros((len(alea),len(mtrans)+1,len(imagen)))
PSNR = np.zeros((len(alea),len(mtrans)+1,len(imagen)))
SSIM = np.zeros((len(alea),len(mtrans)+1,len(imagen)))


# Constantes a usar
rm=0 # valor medio ruido blanco
valpha=1e-6

for k2 in range(0,len(imagen)): 
    # Generación del vector de mediciones
    im = imagen[k2]
    if im == 'SL':
        P0 = shepp_logan(nx)
    else:
        P0 = phantomfun(im) # presión inicial normalizada
    P0 = P0.ravel()
    Pd = Ao@P0
    
    # Agregado de ruido
    #rm=0 # valor medio ruido blanco
    rstd=0.01*np.max(np.abs(Pd)) # desviación estandar ruido 
    ruido=np.random.normal(rm,rstd,(Ns*Nt,))
    
    Pd = Pd + ruido
    
    # RP
    #print('Recontruccion usando RP...');
    trp = np.reshape(tproc,(1,Nt))
    Pd = np.reshape(Pd,(Ns,Nt))

    P0rp = BPv1(nx,dx,vs,cr,arco,trp,Pd) # Filtered BP
        
    P0=np.reshape(P0,(nx,nx));
    P0rp=np.reshape(P0rp,(nx,nx));
    
    cont = 0
    for k1 in range(0,len(alea)):
        # CS
        #print('Recontruccion usando CS...');
        if alea[k1] == True:
            if cont == 0:
                fNtr = 0.7
                Nt=int(Nt/fNtr); # cantidad de muestras
                tproc=np.linspace(to,tf,Nt);
                # Distribución aleatoria de angulos medidos
                Angulos = np.linspace(0,arco*np.pi/180,arco); 
                idx2 = np.random.permutation(np.arange(0,len(Angulos),dtype=int));
                aS = idx2[0:Ns]
                th2 = Angulos[aS];
                # Distribucion uniforme de angulos medidos
                #th2 = th
                Ao=mbt2DLSv5(nx,dx,vs,Ns,cr,arco,ls,nls,tproc,th2);
                P0 = P0.ravel()
                Pd = Ao@P0
                # Agregado de ruido
                rm=0 # valor medio ruido blanco
                rstd=0.01*np.max(np.abs(Pd)) # desviación estandar ruido 
                ruido=np.random.normal(rm,rstd,(Ns*Nt,))
                Pd = Pd + ruido
                P0=np.reshape(P0,(nx,nx));
                Pd = np.reshape(Pd,(Ns,Nt))
            
                tiporand = 'G'
                cont = cont + 1
            else:
                tiporand = 'B'
        elif alea[k1] == False:
            fNtr = 1
            tiporand = 'N'        
        
        for k3 in range(0,len(mtrans)):
            
            print('Imagen: ',imagen[k2], '  Psi: ', mtrans[k3], '  Aleatorio? ',str(alea[k1]), '  Tipo Random: ',tiporand)
                  
            mtxPsi=mtrans[k3]
            #valpha=1e-6
            P0l1=cstv1(Ao,Pd,mtxPsi,alea[k1],tiporand,valpha,fNtr)
    
            P0l1=np.reshape(P0l1,(nx,nx));
            #P0=np.reshape(P0,(nx,nx));
            
            # Calculo de Pearson Correlation
            PC[k1,k3,k2]=stats.pearsonr(P0.ravel(),P0l1.ravel())[0] #np.corrcoef(P0l1.ravel(),P0.ravel())[0, 1]
    
            # Calculo de error cuadrático medio (RMSE)
            RMSE[k1,k3,k2]=math.sqrt(mean_squared_error(P0,P0l1))
    
            # Calculo de relación señal a ruido pico
            PSNR[k1,k3,k2]=peak_signal_noise_ratio(P0,P0l1)
    
            # Calculo del indice de similiridad de estructura 
            SSIM[k1,k3,k2]=structural_similarity(P0,P0l1)
            
                   
        # Metricas para DAS
        PC[k1,-1,k2]=stats.pearsonr(P0.ravel(),P0rp.ravel())[0]  #np.corrcoef(P0rp.ravel(),P0.ravel())[0, 1]
        RMSE[k1,-1,k2]=math.sqrt(mean_squared_error(P0,P0rp))
        PSNR[k1,-1,k2]=peak_signal_noise_ratio(P0,P0rp)
        SSIM[k1,-1,k2]=structural_similarity(P0,P0rp)
    
