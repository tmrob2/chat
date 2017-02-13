# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:13:50 2017

@author: 611004435
"""
import math
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import deque



class Individuals():
    
    def __init__(self, arrival_time, id):
        self.arrival_time = arrival_time
        self.id = id
        self.wait_time = 0.
        self.departure_time = 0.
        
        
class SimulationModel():
    
    def __init__(self, service_rate, server_count):
        self.queue = deque()
        self.demand = self.generate_demand(18) 
        self.history = {}
        self.service_rate = service_rate
        self.server_count = server_count
        
    def MMS1PS_simulation_loop(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):
        t, na, c1, c2 = 0, 0, 0, 0

        # generate a two server model
        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(self.server_count)]
        # generate the first service time
        t0 = rnd.expovariate(self.demand[5])*60.
        ta = t0

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]
            # this could be zero length for which the person will be assigned to a queue and given a wait time
            
            if model_type == 'mbf':
                agents.sort(key=operator.itemgetter(0), reverse=True)
            else: 
                agents.sort(key=operator.itemgetter(0))

            min_agent_ts = min([x.min_service_time() if len(x.service) != 0 else math.inf for x in shift])

            if ta == min(min_agent_ts,ta):
                # add another individual
                na += 1
                t = ta
                ind = Individuals(t, na)
                self.history[ind.id] = ind

                if int(ta/60) < 5 or int(1080 - ta/60) < 5:
                    ta = rnd.expovariate(self.demand[5])*600. + t
                else:
                    ta = rnd.expovariate(self.demand[int(t)])*600. + t

                if len(agents) != 0:
                    # An agent has capacity to serve a customer
                    if len(self.queue) > 0:
                        ind_to_serve = self.queue.popleft()
                        self.queue.append(ind)
                    else:
                        ind_to_serve = ind
                    shift[agents[0][1]].add_to_queue(ind_to_serve, t)
                    for i in self.queue:
                        i.wait_time = ta - i.arrival_time
                else:
                    # An agent does not have capacity to serve a customer
                    self.queue.append(ind)
                    for i in self.queue:
                        i.wait_time = ta - i.arrival_time
            else:
                
                # Ammend the system time
                t = min_agent_ts
                for i in shift:
                    i.depart_customer_from_service_line(t)
                for i in self.queue:
                    i.wait_time = t - i.arrival_time

        while len(self.queue) != 0:

            # this could be zero length for which the person will be assigned to a queue and given a wait time
            min_agent_ts = min([x.min_service_time() if len(x.service) != 0 else math.inf for x in shift])
            t = min_agent_ts
            for i in self.queue:
                i.wait_time = t - i.arrival_time

            ind_to_serve = self.queue.popleft()
            for i in shift:
                i.depart_customer_from_service_line(t)

            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]

            if model_type == 'mbf':
                agents.sort(key=operator.itemgetter(0), reverse=True)
            else:
                agents.sort(key=operator.itemgetter(0))

            shift[agents[0][1]].add_to_queue(ind_to_serve, t)



        return shift, self.history

    def generate_demand(self, work_day_hours: int):
        """
        Generates a workday expectation of volume offered to servers
        Returns a list of demand of discrete time intervals (minutes)
        is a concave quadratic continuous function in the R2 space
        """
        t = np.linspace(0,work_day_hours,work_day_hours*60*60)
        demand = -3*t*(t-work_day_hours)
        return demand

    def plot_wait_time(self):
        wt = []
        for i in range(1,len(self.history)):
            if self.history[i].wait_time != 0:
                wt.append(self.history[i].wait_time)
        plt.figure()
        plt.hist(wt, 500)
        plt.xlim(0,500)
        return 1

    def plot_service_time(self):
        st = []
        dt = []
        for i in range(1, len(self.history)):
            st.append(self.history[i].arrival_time)
            dt.append(self.history[i].departure_time)

        plt.figure()
        plt.hist(st, 50)
        plt.title('Arrival Time')

        plt.figure()
        plt.hist(dt,50)
        plt.title('Departure Time')


class Servers:

    def __init__(self, mean_ts, id, conc_lim):
        self.arrival_times = []
        self.customer_number = []
        self.departure_times = []
        self.server_times = []
        self.server_queue = []
        self.service = {} #this is the operating service and limited by the concurrent chats pair(Individual, departure_time)
        self.idle_time = []
        self.concurrent_chats = 0
        self.in_service_ts = []
        self.mean_service_time = mean_ts
        self.id = id
        self.concurrency_measure = []
        self.conc_lim = conc_lim

    def add_to_queue(self, ind: Individuals, t):
        self.server_queue.append(ind)
        
        if self.concurrent_chats <= self.conc_lim:
            self.add_to_service_line(ind, t)
            
    def add_to_service_line(self, ind: Individuals, t):
        t_exp = rnd.expovariate(1 / self.mean_service_time)*60.
        ts = t + t_exp
        self.server_times.append(t_exp)
        ind.departure_time = ts
        self.service[ind.id] = {'ta': ind.arrival_time,
                                'ts': ts}
        self.concurrent_chats += 1

    def min_service_time(self):
        m = math.inf
        for k,v in self.service.items():
            if v['ts'] < m:
                m = v['ts']
        return m

    def depart_customer_from_service_line(self,t):
        self.service = {k:v for k,v in self.service.items() if v['ts'] != t}
        self.concurrent_chats = len(self.service)




        
        
        
        