#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:39:30 2021

@author: martin
"""

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from scipy import stats
import math

def phantomfun(imagen):
    '''
    Funcion que entrega una imagen en escala de grises de los siguientes fantomas de nx x nx pixeles:
    TOAnx: permite analizar la performance para recuperar objetos con bordes agudos.
    Derenzonx: permite determinar la fortaleza del métodos ante objetos grandes y pequeños.
    bloodvesselnx: permite analizar la performance ante estructura amorfas complicadas.
    breastnx: permite analizar la robustez en casos complejos donde también el contraste varía.
    '''
    pathfile='Fantomas/'
    tifile=pathfile+imagen+'.tif'
    im=io.imread(tifile)
    im=rgb2gray(im)
    im=im/np.max(im.ravel())
    im=im*-1+1
    #plt.figure(30)
    #plt.imshow(im)
    fantoma=im
    return(fantoma)
