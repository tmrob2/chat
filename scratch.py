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
import scipy.stats 
import pandas as pd

a = []
b = []
c = []
d = []
e = []

for i in range(0,100000):
    #a.append(random.expovariate(demand[50])*60.)
    #b.append(random.expovariate(demand[540])*60.)
    #c.append(random.expovariate(demand[5])*60.)
    d.append(random.expovariate(1/25)*60.)
    e.append(random.weibullvariate(math.exp(7.111),1/0.902))

fig = plt.figure()
n, x1, _ = plt.hist(a, 500, normed=1, alpha=0.75)
density1 = scipy.stats.gaussian_kde(a)
fig = plt.figure()
n, x2, _ = plt.hist(b, 500, normed=1, alpha=0.75)
density2 = scipy.stats.gaussian_kde(b)
fig = plt.figure()
n, x3, _ = plt.hist(c, 500, normed=1, alpha=0.75)
density3 = scipy.stats.gaussian_kde(b)
fig = plt.figure()
plt.hist(d, 500, normed=1, alpha=0.75)
fig = plt.figure()
plt.hist(e, 100, normed=1, alpha=0.75)

fig = plt.figure()
plt.plot(x1, density1(x1), '-r', linewidth = 1.5)
plt.plot(x2, density2(x2), '-k', linewidth = 1.5)
plt.plot(x3, density2(x2), '-b', linewidth = 1.5)

t = np.linspace(0,18,18*60)
demand = -7*t*(t-18)
plt.figure()
plt.plot(t,demand)

p=1.2
p=1.4
lam = 1
t = np.linspace(1,1000,1000)
survival = np.exp(-np.exp(-7.5/0.902)*t**(1/0.902))
plt.figure()
plt.plot(t, survival)


theta = np.linspace(1/50, 2,10000)
def eps(k, d, mu):
    if k == 0:
        return 1
    else:
        return 1+(k*mu)/d *eps(k-1, d, mu) 

fig = plt.figure()


d = demand[5]/3600
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

import scipy.stats 
import pandas as pd
import MMnPSQ
sim = MMnPSQ.SimulationModel(30,400)
Sims = [sim.MMS1PS_simulation_loop_multisim(64800,3) for i in range(10)]
ls = []
for i in range(0, len(Sims)):
    wt,st,ar,agents,ans = Sims[i]
    d={'wait': wt, 'service': st, 'aband': ar, 'servers': agents, 'answered': ans}
    ls.append(d)
df5 = pd.DataFrame(ls)
df5.to_csv("df_400.csv")
print(df5)


#single sim results
import pandas as pd
import MMnPSQ
sim = MMnPSQ.SimulationModel(30,180)
shift, hist = sim.MMS1PS_simulation_loop_singlesim(64800,3)