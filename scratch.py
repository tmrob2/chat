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
demand = -15*t*(t-18)
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
sim = MMnPSQ.SimulationModel(20,180)
Sims = [sim.MMS1PS_simulation_loop_multisim(57600,2) for i in range(3)]
ls = []
for i in range(0, len(Sims)):
    avg_wait_time, median_wait_time, max_wait_time, avg_service_time, avg_ar, sum_chats_answered = Sims[i]
    d={'wait': avg_wait_time,'med_wait': median_wait_time,'max_wait': max_wait_time, 
       'service': avg_service_time, 'aband': avg_ar, 'answered': sum_chats_answered}
    ls.append(d)
df1 = pd.DataFrame(ls)
df1.to_csv("sim_highvol_20perc_fte_saving.csv")
print(df1)


# Single queue multiple servers cross skilled
import pandas as pd
import MMnPSQ
sim = MMnPSQ.SimulationModel(20,180)
shift, hist = sim.MMS1PS_simulation_loop_singlesim(57600,2)
sim.plot_abandoned("highvol_sqms_xskill")
sim.plot_service_time("highvol_sqms_xskill_arrival_times", "mqms_depart_times")
sim.plot_idle_hist(shift,"highvol_sqms_xskill_agent_idle")
sim.plot_wait_time("highvol_sqms_xskill_cust_wait")
print('finished')
# Multi server with queues cross skilled
import MMnnPSQ
sim = MMnnPSQ.SimulationModel(20,180)
shift, hist = sim.MMSNPS_simulation_loop_singlesim(57600, 2)
sim.plot_abandoned("mqms_xskill_q")
sim.plot_service_time("mqms_xskill_arrival_times", "mqms_depart_times")
sim.plot_idle_hist(shift,"mqms_xskill_agent_idle")
sim.plot_wait_time("mqms_xskill_cust_wait")
print('finished')

import pandas as pd
test = pd.read_csv('overallshift.csv')
t0 = 0.0
test.query('start <= %s <= end'%0.0)['Mon'].values[0]

# multi simulation 
import scipy.stats 
import pandas as pd
import MMnnPSQ
sim = MMnnPSQ.SimulationModel(20,180)
Sims = [sim.MMSNPS_simulation_loop_multisim(57600,2) for i in range(3)]
ls = []
for i in range(0, len(Sims)):
    avg_wait_time, median_wait_time, max_wait_time, avg_service_time, avg_ar, sum_chats_answered = Sims[i]
    d={'wait': avg_wait_time,'med_wait': median_wait_time,'max_wait': max_wait_time, 
       'service': avg_service_time, 'aband': avg_ar, 'answered': sum_chats_answered}
    ls.append(d)
df1 = pd.DataFrame(ls)
df1.to_csv("df_180.csv")
print(df1)

r = random.uniform(0,1)
p = [0.,0.354,0.654,0.797,0.897,0.976,1.]

# multiple service queues with agent queues not cross skilled
import MMnnPSQ
sim = MMnnPSQ.SimulationModel(20,180)
shift, hist = sim.MMSNPS_simulation_loop_singlesim_seg_qs(57600, 2)
sim.plot_abandoned("mqms_teams")
sim.plot_service_time("mqms_teams_arrival_times", "mqms_depart_times")
sim.plot_idle_hist(shift,"mqms_teams_agent_idle")
sim.plot_wait_time("mqms_teams_cust_wait")
print('finished')

import scipy.stats 
import pandas as pd
import MMnnPSQ
sim = MMnnPSQ.SimulationModel(20,180)
Sims = [sim.MMSNPS_simulation_loop_multisim_seg_qs(57600,2) for i in range(50)]
ls = []
for i in range(0, len(Sims)):
    avg_wait_time, median_wait_time, max_wait_time, avg_service_time, avg_ar, sum_chats_answered = Sims[i]
    d={'wait': avg_wait_time,'med_wait': median_wait_time,'max_wait': max_wait_time, 
       'service': avg_service_time, 'aband': avg_ar, 'answered': sum_chats_answered}
    ls.append(d)
df1 = pd.DataFrame(ls)
df1.to_csv("df_140x.csv")
print(df1)
