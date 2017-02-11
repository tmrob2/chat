# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:09:19 2017

@author: 611004435
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
import math
import mpmath
from scipy.stats import norm


a = []
b = []

for i in range(0,1000):
    a.append(random.expovariate(1/25))
    b.append(random.betavariate(5,5))

#plt.hist(a, 50, normed=1, facecolor='green', alpha=0.75)
plt.hist(b, 50, normed=1, faceolor = 'blue', alpha=0.75)
plt.show()

t = np.linspace(0,18,18*60)
demand = -6*t*(t-18)


theta = np.linspace(1/50, 2,10000)
def eps(k, d, mu):
    if k == 0:
        return 1
    else:
        return 1+(k*mu)/d *eps(k-1, d, mu) 

fig = plt.figure()


d = demand[540]/3600
mu = 1/(30*60)
n=200
c = (n*mu)/theta
x = d/theta
J = np.zeros(len(theta))
for i in range(0,len(theta)): 
    J[i] = (math.exp(d/theta[i])/theta[i]) * (theta[i]/d)**(c[i])*(mpmath.gammainc(c[i],x[i])) 
##plt.plot(theta, J)
P = (1 + (d - n*mu)*J)/(eps(n,d,mu)+d*J)
##fig2 = plt.figure()
plt.plot(theta,P)
plt.ylim(0,1)
plt.xlim(0.05,1)
