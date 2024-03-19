#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:42:08 2024

@author: Estudiante
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1,1,1],[4,2,1],[9,3,1]])
b = np.array([1,2,0])

a,b,c = np.linalg.solve(A,b)

xx = np.array([1,2,3])
yy = np.array([1,2,0])
x = np.linspace(0,4,100) #genera 100 puntos equiespaciados entre 0 y 4.
f = lambda t: a*t**2 + b*t + c #estogeneraunafuncionfdet.
plt.plot(xx,yy,'*')
plt.plot(x,f(x))
plt.show()