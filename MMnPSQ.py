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
        self.abandoned = 0
        
        
class SimulationModel():
    
    def __init__(self, service_rate, server_count):
        self.queue = deque()
        self.demand = self.generate_demand(18) 
        self.history = {}
        self.service_rate = service_rate
        self.server_count = server_count
        t = np.linspace(1,18*60*60,18*60*60)
        self.survival = np.exp(-np.exp(-11.5/0.902)*t**(1/0.902))
        
    def MMS1PS_simulation_loop_multisim(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):
        t, na, c1, c2 = 0, 0, 0, 0

        # generate a two server model
        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(self.server_count)]
        # generate the first service time
        t0 = rnd.expovariate(self.demand[5])*60.
        ta = t0

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]

            for i in shift:
                i.generate_idle_flag(t)
            
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
                    ls_ind_abd = []
                    for i in self.queue:
                        i.wait_time = t - i.arrival_time
                        if self.generate_abandoment(i.wait_time) == True:
                            ls_ind_abd.append(i)
                            i.abandoned = 1
                    for i in ls_ind_abd:
                        self.queue.remove(i)
                else:
                    # An agent does not have capacity to serve a customer
                    self.queue.append(ind)
                    ls_ind_abd = []
                    for i in self.queue:
                        i.wait_time = t - i.arrival_time
                        if self.generate_abandoment(i.wait_time) == True:
                            ls_ind_abd.append(i)
                            i.abandoned = 1
                    for i in ls_ind_abd:
                        self.queue.remove(i)
            else:
                
                # Ammend the system time
                t = min_agent_ts
                for i in shift:
                    i.depart_customer_from_service_line(t)
                ls_ind_abd = []
                for i in self.queue:
                    i.wait_time = t - i.arrival_time
                    if self.generate_abandoment(i.wait_time) == True:
                        ls_ind_abd.append(i)
                        i.abandoned = 1
                for i in ls_ind_abd:
                    self.queue.remove(i)

        while len(self.queue) != 0:

            # this could be zero length for which the person will be assigned to a queue and given a wait time
            min_agent_ts = min([x.min_service_time() if len(x.service) != 0 else math.inf for x in shift])
            t = min_agent_ts
            ls_ind_abd = []
            for i in self.queue:
                i.wait_time = t - i.arrival_time
                if self.generate_abandoment(i.wait_time) == True:
                    ls_ind_abd.append(i)
                    i.abandoned = 1
            for i in ls_ind_abd:
                self.queue.remove(i)

            ind_to_serve = self.queue.popleft()
            for i in shift:
                i.depart_customer_from_service_line(t)

            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]

            if model_type == 'mbf':
                agents.sort(key=operator.itemgetter(0), reverse=True)
            else:
                agents.sort(key=operator.itemgetter(0))

            shift[agents[0][1]].add_to_queue(ind_to_serve, t)

        avg_wait_time = sum([self.history[i].wait_time for i in range(1,len(self.history))]) / len(self.history)
        avg_service_time = sum([sum(i.server_times)/len(i.server_times) for i in shift])/len(shift)
        sum_ar = sum([self.history[i].abandoned for i in range(1,len(self.history))])/len([self.history[i].abandoned for i in range(1,len(self.history)) if self.history[i].abandoned == 0])
        sum_chats_answered = sum([1 for i in range(1,len(self.history)) if self.history[i].abandoned == 0])
        #return shift, self.history
        return avg_wait_time, avg_service_time, sum_ar, self.server_count, sum_chats_answered

    def MMS1PS_simulation_loop_singlesim(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):
        t, na, c1, c2 = 0, 0, 0, 0

        # generate a two server model
        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(self.server_count)]
        # generate the first service time
        t0 = rnd.expovariate(self.demand[5])*60.
        ta = t0

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]

            for i in shift:
                i.generate_idle_flag(t)
            
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
                    ls_ind_abd = []
                    for i in self.queue:
                        i.wait_time = t - i.arrival_time
                        if self.generate_abandoment(i.wait_time) == True:
                            ls_ind_abd.append(i)
                            i.abandoned = 1
                    for i in ls_ind_abd:
                        self.queue.remove(i)
                else:
                    # An agent does not have capacity to serve a customer
                    self.queue.append(ind)
                    ls_ind_abd = []
                    for i in self.queue:
                        i.wait_time = t - i.arrival_time
                        if self.generate_abandoment(i.wait_time) == True:
                            ls_ind_abd.append(i)
                            i.abandoned = 1
                    for i in ls_ind_abd:
                        self.queue.remove(i)
            else:
                
                # Ammend the system time
                t = min_agent_ts
                for i in shift:
                    i.depart_customer_from_service_line(t)
                ls_ind_abd = []
                for i in self.queue:
                    i.wait_time = t - i.arrival_time
                    if self.generate_abandoment(i.wait_time) == True:
                        ls_ind_abd.append(i)
                        i.abandoned = 1
                for i in ls_ind_abd:
                    self.queue.remove(i)

        while len(self.queue) != 0:

            # this could be zero length for which the person will be assigned to a queue and given a wait time
            min_agent_ts = min([x.min_service_time() if len(x.service) != 0 else math.inf for x in shift])
            t = min_agent_ts
            ls_ind_abd = []
            for i in self.queue:
                i.wait_time = t - i.arrival_time
                if self.generate_abandoment(i.wait_time) == True:
                    ls_ind_abd.append(i)
                    i.abandoned = 1
            for i in ls_ind_abd:
                self.queue.remove(i)

            ind_to_serve = self.queue.popleft()
            for i in shift:
                i.depart_customer_from_service_line(t)

            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]

            if model_type == 'mbf':
                agents.sort(key=operator.itemgetter(0), reverse=True)
            else:
                agents.sort(key=operator.itemgetter(0))

            shift[agents[0][1]].add_to_queue(ind_to_serve, t)

        avg_wait_time = sum([self.history[i].wait_time for i in range(1,len(self.history))]) / len(self.history)
        avg_service_time = sum([sum(i.server_times)/len(i.server_times) for i in shift])/len(shift)
        sum_ar = sum([self.history[i].abandoned for i in range(1,len(self.history))])/len([self.history[i].abandoned for i in range(1,len(self.history)) if self.history[i].abandoned == 0])
        return shift, self.history
        #return avg_wait_time, avg_service_time, sum_ar, self.server_count

    def generate_demand(self, work_day_hours: int):
        """
        Generates a workday expectation of volume offered to servers
        Returns a list of demand of discrete time intervals (minutes)
        is a concave quadratic continuous function in the R2 space
        """
        t = np.linspace(0,work_day_hours,work_day_hours*60*60)
        demand = -7*t*(t-work_day_hours)
        return demand

    def generate_abandoment(self, wait_time):
        p = rnd.uniform(0,1)
        if p > self.survival[int(wait_time)]:
            return True
        else:
            return False

    def plot_wait_time(self):
        wt = []
        for i in range(1,len(self.history)):
            if self.history[i].wait_time != 0:
                wt.append(self.history[i].wait_time)
        plt.figure()
        plt.hist(wt, 100)
        plt.xlim(0,100)
        return 1

    def plot_service_time(self):
        st = []
        dt = []
        for i in range(1, len(self.history)):
            st.append(self.history[i].arrival_time)
            if self.history[i].departure_time != 0:
                dt.append(self.history[i].departure_time)

        plt.figure()
        plt.hist(st, 50)
        plt.title('Arrival Time')
        plt.savefig("arrival_time_sim_output.png")

        plt.figure()
        plt.hist(dt,50)
        plt.title('Departure Time')
        plt.savefig("departure_time_sim_output.png")

    def plot_abandoned(self):
        ab = []
        not_ab = []
        for i in range(1, len(self.history)):
            if self.history[i].abandoned == 1:
                ab.append(self.history[i].arrival_time)
            else:
                not_ab.append(self.history[i].arrival_time)

        plt.figure()
        p1 = plt.hist([ab, not_ab],50,stacked = True)
        plt.title('Abandonment Count')
        plt.savefig("sim_abandonment_rate.png")

    def plot_idle_hist(self, shift):
        idle = []
        for i in shift:
            for j in i.idle_time:
                idle.append(j)
        plt.figure()
        bins = 50
        n, x, _ = plt.hist(idle, bins, normed=1, alpha=0.75)
        plt.figure()
        plt.plot(x[:-1], n*bins*self.server_count)
        plt.title('Count of agents idle at time t')
        plt.savefig("staff_idle_hours.png")
        return n, x

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

    def generate_idle_flag(self, t):
        if self.concurrent_chats == 0:
            self.idle_time.append(int(t/60))







        
        
        
        