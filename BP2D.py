#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:31:44 2021

Last updated on Jun 6 18:32:00 2021

@author: martu
"""

# Módulos
#from pytictoc import TicToc
import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d
from tqdm import tqdm # barra de progreso para loops

# Funciones auxilares 
def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int64") // array_shape[1])
    cols = (ind.astype("int64") % array_shape[1])
    return (rows, cols)

def DASv1(nx,dx,vs,cr,arco,t,p):
    """
    Esta función realiza la reconstruccion de una imagen a partir de señales OA.
    El algoritmo sigue el enfoque retroproyección (BP) básico también llamado
    "Delay and Sum" (DAS), donde los conceptos básico se pueden leer en:
    X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming 
    Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019).
    La salida es un vector imagen [nx*nx x 1]
    
    La sintaxis completa es la siguiente:

          A=DASv1(nx,dx,vs,cr,arco,t,p)

    A continuación se detallan los parámetros de entrada: 
    nx: cantidad de puntos de grilla cuadrada uniforme (ny=nx).
    dx: espaciado entre puntos de grilla (dy=dx).
    vs: velocidad del sonido en el medio [m/s].
    cr: radio de la circumferencia donde se colocan los sensores [m].
    arco: arco de circunferencia donde están dispuestos los sensores [grados].
    t: vector de tiempos [s] de tamaño [1xNt] donde Nt es cantidad de muestras.
    p: matriz de presiones o potencial de velocidades de tamaño [Ns x Nt]. 
       donde Ns es cantidad de sensores
    """
    
    #print('-'*30)
    #print('Por favor espere, estamos reconstruyendo la imagen OA...');
    #print('-'*30)
    
    # GRILLA
    ny = nx;   # nx = ny. Se diferencian por si se quiere implemetnar algo a futuro de grilla no uniforme y/o rectangular. 
    dy = dx;   # dx, dy, dz = espaciado entre puntos de grilla. dx = dy. Idem arriba
        
    # Tamaños de las mediciones e imagen a reconstruir
    Nt=np.size(t,1) # cantidad de muestras temporales
    Ns=np.size(p,0) # cantidad de sensores o angulos (en el caso de un sensor)
    N = nx*ny; # Cantidad total de elementos de la grilla
        
    #print('-'*30)
    #print('Parametros relevantes: muestras de tiempo, cantidad de sensores y tamaño grilla: ')
    #print('Nt: ',Nt)
    #print('Ns :',Ns)
    #print('Nx :',N)
    #print('-'*30)
    
    # Coordenadas GRILLA IMAGEN
    originX = np.ceil(nx/2); originY = np.ceil(ny/2); # Defino el origen de coordenadas de la grilla
    y,x = ind2sub([nx,ny], np.linspace(0,N-1,N)); # Devuelve la fila y columna en la grilla discretizada de tamaño [nx,ny] correspondiente al indice lineal j.
    # Lo hago para saber la posicion del elemento de volumen en la grilla.
    # Coordenadas de cada elemento de la grilla referido al centro de la misma (lo cual defino como origen con coordeandas (0,0))
    rj=np.array([(x-originX)*dx,(y-originY)*dy])
    rj=np.transpose(rj)
    Rj=np.reshape(rj,(1,N*2))
    Rj=np.repeat(Rj,Ns,axis=0)
    Rj=np.reshape(Rj,(Ns*N*2,1))
        
    #POSICION DE LOS SENSORES
    # Cálculo el vector de posición de cada elemento de cada sensor referido al origen de coordenadas
    # Centro de los sensores, definidos sobre una circumferencia
    th = np.linspace(0,arco*np.pi/180,Ns+1); th = th[0:len(th)-1];
    rs=np.array([cr*np.cos(th),cr*np.sin(th)])
    rs=np.transpose(rs)
    Rs=np.repeat(rs,N,axis=0)
    Rs=np.reshape(Rs,(Ns*N*2,1))
 
    # Tiempo relacionado con los puntos de la GRILLA
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # función que calcula la norma 2 por fila
    Tau=np.reshape(Tau,(Ns,N))
    
    # Determinación de la imagen
    DAS=np.zeros((N,))
    for i in tqdm(range(0,Ns)):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        DAS=DAS+aux
    
    # Para que sea compatible con las otras funciones que reconstruyen (pej. lsqr), la salida es un vector de [N,] 
    #DAS=np.reshape(DAS,(nx,nx)) 
    return DAS 

def BPv1(nx,dx,vs,cr,arco,t,p):
    """
    Esta función realiza la reconstruccion de una imagen a partir de señales OA.
    El algoritmo sigue el enfoque retroproyección (BP) básico.
    La salida es un vector imagen [nx*nx x 1]
    
    La sintaxis completa es la siguiente:

          A=RPv1(nx,dx,vs,cr,arco,t,p)

    A continuación se detallan los parámetros de entrada: 
    nx: cantidad de puntos de grilla cuadrada uniforme (ny=nx).
    dx: espaciado entre puntos de grilla (dy=dx).
    vs: velocidad del sonido en el medio [m/s].
    cr: radio de la circumferencia donde se colocan los sensores [m].
    arco: arco de circunferencia donde están dispuestos los sensores [grados].
    t: vector de tiempos [s] de tamaño [1xNt] donde Nt es cantidad de muestras.
    p: matriz de presiones o potencial de velocidades de tamaño [Ns x Nt]. 
       donde Ns es cantidad de sensores
    """
    
    #print('-'*30)
    #print('Por favor espere, estamos reconstruyendo la imagen OA...');
    #print('-'*30)
      
    # GRILLA
    ny = nx;   # nx = ny. Se diferencian por si se quiere implemetnar algo a futuro de grilla no uniforme y/o rectangular. 
    dy = dx;   # dx, dy, dz = espaciado entre puntos de grilla. dx = dy. Idem arriba
        
    # Tamaños de las mediciones e imagen a reconstruir
    Nt=np.size(t,1) # cantidad de muestras temporales
    Ns=np.size(p,0) # cantidad de sensores o angulos (en el caso de un sensor)
    N = nx*ny; # Cantidad total de elementos de la grilla
        
    #print('-'*30)
    #print('Parametros relevantes: muestras de tiempo, cantidad de sensores y tamaño grilla: ')
    #print('Nt: ',Nt)
    #print('Ns :',Ns)
    #print('Nx :',N)
    #print('-'*30)
    
    # Filtrado
    Mt = np.ones((Ns,1))@t
    #p = Mt**np.gradient(p,axis=1)
    p = np.gradient(p,axis=1)
    
    # Coordenadas GRILLA IMAGEN
    originX = np.ceil(nx/2); originY = np.ceil(ny/2); # Defino el origen de coordenadas de la grilla
    y,x = ind2sub([nx,ny], np.linspace(0,N-1,N)); # Devuelve la fila y columna en la grilla discretizada de tamaño [nx,ny] correspondiente al indice lineal j.
    # Lo hago para saber la posicion del elemento de volumen en la grilla.
    # Coordenadas de cada elemento de la grilla referido al centro de la misma (lo cual defino como origen con coordeandas (0,0))
    rj=np.array([(x-originX)*dx,(y-originY)*dy])
    rj=np.transpose(rj)
    Rj=np.reshape(rj,(1,N*2))
    Rj=np.repeat(Rj,Ns,axis=0)
    Rj=np.reshape(Rj,(Ns*N*2,1))
        
    #POSICION DE LOS SENSORES
    # Cálculo el vector de posición de cada elemento de cada sensor referido al origen de coordenadas
    # Centro de los sensores, definidos sobre una circumferencia
    th = np.linspace(0,arco*np.pi/180,Ns+1); th = th[0:len(th)-1];
    rs=np.array([cr*np.cos(th),cr*np.sin(th)])
    rs=np.transpose(rs)
    Rs=np.repeat(rs,N,axis=0)
    Rs=np.reshape(Rs,(Ns*N*2,1))
 
    # Tiempo relacionado con los puntos de la GRILLA
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # función que calcula la norma 2 por fila
    Tau=np.reshape(Tau,(Ns,N))
    
    # Determinación de la imagen
    BP=np.zeros((N,))
    for i in tqdm(range(0,Ns)):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        BP=BP+aux
    
    # Para que sea compatible con las otras funciones que reconstruyen (pej. lsqr), la salida es un vector de [N,] 
    #BP=np.reshape(BP,(nx,nx)) 
    return BP