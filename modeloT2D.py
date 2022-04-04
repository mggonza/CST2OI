#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:33:40 2020

Last updated on Dec 30 19:48:00 2020

@author: martu

"""

# Módulos
from pytictoc import TicToc
import numpy as np
import numpy.matlib
from scipy.linalg import toeplitz
from scipy import sparse
from scipy.sparse import csc_matrix # Sparse por columna
from tqdm import tqdm # barra de progreso para loops

# Funciones auxilares 
def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int64") // array_shape[1])
    cols = (ind.astype("int64") % array_shape[1])
    return (rows, cols)

# Funciones principales:

def mbt2DLSv5(nx,dx,vs,nSens,cr,arco,lz,nDisc,tproc,th):
    """
    Modificación para mediciones experimentales de la version mbt2DLSv4
    """
    #print('Generando modelo directo...');
    # # Calculo de la matriz de potencial de velocidades
    # Contantes
    B=207e-6; # coeficiente expansión volumétrica térmica para agua a 20 °C [1/K]
    Calp=4184; # capacidad calorífica específica promedio para agua a 20 °C [J/K Kg]
    # Cp=4180; # a 35 °C
    #Cp=3.5e3; # capacidad calorífica específica promedio para piel humana [J/K Kg]
    rho=1000; # densidad para agua o tejido blando [kg/m^3]
    #ru=B/Calp*vs^2; % Parámetro adimensional de Gruneisen
    #mua=100; # coeficiente de absorción promedio para tejido blando [1/m]
    #FLa=200; #Energía por unidad de área máxima con la que puede irradiarse en el visible con pulsos < 100 ns 
    h0=1e4; # h0=mua*FLa; energía por unidad de volumen típica entregada a la muestra [J/m^3] 
    phi0=1e-11; # phi0=dt*p0/rho; [m^2/s] donde p0 = 100 Pa (presiones típicas).
    # EJE DE TIEMPO
    #tim=nx*dx; # Tamaño de la imagen tim x tim
    #CFL=0.25; # Número de Courant-Friedrichs-Lev
    dt=tproc[1]-tproc[0];# dt=dt[0];  # Paso de tiempo
    #to=(cr-tim)/vs*0.9; tf=to+(tim/vs)*1.5; # tiempo inicial y final
    to=int(tproc[0]/dt); tf=int(tproc[len(tproc)-1]/dt)+1;
    nSamples=tf-to; # Cantidad de muestras temporales
    sampleTimes = np.arange(to,tf,dtype=int);
    # GRILLA
    ny = nx;   # nx = ny. Se diferencian por si se quiere implemetnar algo a futuro de grilla no uniforme y/o rectangular. 
    dy = dx;   # dx, dy, dz = espaciado entre puntos de grilla. dx = dy. Idem arriba
    dz = dx;   # dz = no se usa. Queda por si se quiere implementar algo a futuro 
    DVol=dx*dy*dz; # volumen de un elemento de grilla
    N = nx*ny; # Cantidad total de elementos de la grilla
    #print('Nt:',nSamples)
    #print('Ns:',nSens)
    #print('Nx:',N)
    originX = np.ceil(nx/2); originY = np.ceil(ny/2); # Defino el origen de coordenadas de la grilla
    y,x = ind2sub([nx,ny], np.linspace(0,N-1,N)); # Devuelve la fila y columna en la grilla discretizada de tamaño [nx,ny] correspondiente al indice lineal j.
    # Lo hago para saber la posicion del elemento de volumen en la grilla.
    # Coordenadas de cada elemento de la grilla referido al centro de la misma (lo cual defino como origen con coordeandas (0,0,0))
    rj=np.array([(x-originX)*dx,(y-originY)*dy,np.zeros((len(x)))])
    #POSICION DE LOS SENSORES Y/O DISCRETIZACION SENSOR LINEAL
    # Cálculo el vector de posición de cada elemento de cada sensor referido al origen de coordenadas
    # posSens posición de los centros de los sensores ubicados sobre una circumferencia que rodea el plano de la muestra. 
    # Es decir, cada elemento de posSens es de la forma [x y 0], suponiendo que la muestra esta en el plano XY (z=0).
    # Centro de los sensores, definidos sobre una circumferencia
    #th = np.linspace(0,arco*np.pi/180,nSens+1); th = th[0:len(th)-1]; # Angulos
    posSens=np.array([cr*np.cos(th),cr*np.sin(th),np.zeros((len(th)))])# Centro de los sensores lineales
    #posSens=posSens*(1+ipos/100); # Agregado de incerteza
    # posZ contiene la posición z de cada sensor discreto, los cuales esta ubicados por "arriba" y por
    # "abajo" de los centros de los sensores lineal. Las posiciones z son las mismas para todos los sensores,
    # lo único que cambia es donde estan ubicados sobre el plano x,y
    # posSensLin(:,:,i) contiene todas las posiciones de los sensores discretos que conforman al sensor lineal i
    posSensLin = posSens; 
    if nDisc>3:
        posz = np.linspace(-lz/2, lz/2, nDisc); posz=np.reshape(posz,(len(posz),1)); # coordenada z de los sensores
        posSensLin = np.reshape(posSens,(1,3,nSens));
        aux=posSensLin;
        for k in range(1,len(posz)):
            posSensLin = np.vstack((posSensLin,aux));
        posSensLin[0:,2,0:] = np.matlib.repmat(posz,1,nSens);
    else:
        posz = 0; posz=np.array([posz]);
        posSensLin = np.reshape(posSens,(1,3,nSens)); 
        posSensLin[0:,2,0:] = np.matlib.repmat(posz,1,nSens);
    # MATRIZ DE SENSADO
    # El procesamiento de A se hace por filas porque por razones internas de como organiza la memoria en los arrgelos Python resulta más rapido y eficiente
    nSamples=len(sampleTimes);
    A = np.zeros((nSamples*nSens,N)); 
    currentSens = 0;    # Contador para el sensor actual        
    currentTime = 0;    # Contador para el tiempo actual
    # Ciclo para calcular los coeficientes de A
    for i in tqdm(range(1,np.size(A,0)+1)): #Filas
        acum = np.zeros((1,np.size(A,1))); # Acá se guarda el resultado de cada sensor puntual discreto que compone al sensor lineal extenso.
        for kk in range(0,nDisc): # Para cada elemento discreto que compone al sensor lineal actual (currentSens)
            # Cálculo la norma de la resta entre la posicion del elemento de volumen y elemento discreto del sensor actual (R = |posSens - r_j|)
            aux=np.reshape(posSensLin[kk,0:,currentSens],(3,1))@np.ones((1,N));
            aux2=rj-aux;  
            R=np.sqrt(aux2[0,0:]**2+aux2[1,0:]**2+aux2[2,0:]**2); R=np.reshape(R,(1,len(R)));
            # delta = 1   si    |t_k - R/vs| < dt/2
            #         0         en otro caso
            delta = (np.abs(sampleTimes[currentTime]*dt - R/vs) < dt/2); # Calculo el termino de la delta
            delta=delta*1; # Paso de bool a int
            acum = acum + delta/R; # Lo sumo al acumulador
        A[i-1,0:] = acum;
        currentTime = currentTime + 1; # Cuando paso a la siguiente fila, estoy en la siguiente muestra
        # Cuando termino un bloque de tamaño nSamples significa que pase al siguiente sensor.
        # Es decir, cuando el número de fila es múltiplo de nSamples.
        # Tengo que incrementar el numero de sensor y reiniciar el contador de tiempo
        if np.mod(i,nSamples) == 0:
            currentSens = currentSens + 1; 
            currentTime = 0; 
    A = (-B/(4*np.pi*rho*Calp)*DVol/dt)*A;  # Unidades de A son [m^5/(J*s)]
    # Normalización de la matriz A usando valores típicos de energía absorbida por unidad de volumen (h0) y potencial de velocidades phi0 
    # phi0*Phi* = A x h0*H*
    A= -1*A*h0/phi0; 
    A = A/np.max(np.abs(A.ravel())); # Normaliza para que el valor maximo sea 1
    # Para hacer más sparse la matriz
    # umbral=1e-6;
    # indumbral=np.where(A<1e-6);
    # A[indumbral]=0;
    return(A)
