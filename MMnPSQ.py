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


class Individuals():
    
    def __init__(self, arrival_time, id):
        self.arrival_time = arrival_time
        self.id = id
        self.wait_time = 0.
        self.departure_time = 0.
        
        
class SimulationModel():
    
    def __init__(self, service_rate, server_count):
        self.queue = []
        self.demand = self.generate_demand(18) 
        self.history = []
        self.service_rate = service_rate
        self.server_count = server_count
        
    def MMS1PS_simulation_loop(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):
        t,Na,C1,C2 = 0,0,0,0
        SS = np.array([0,0,0])

        #generate a two server model
        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(self.server_count)]
        #generate the first service time
        T0 = rnd.expovariate(self.demand[5])
        while t < max_sim_time:
            ta = T0
           
            #loop through the servers to see who is the most busy but not reached their concurrency limit
            agents = [(x.concurrent_chats, x.id) for x in shift if x.concurrent_chats < concurency_limit]
            #this could be zero length for which the person will be assigned to a queue and given a wait time
            
            if model_type == 'mbf':
                agents.sort(key = operator.itemgetter(0), reverse=True)
            else: 
                agents.sort(key = operator.itemgetter(0))

            min_agent_ts = min([x.min_service_time() if len(x.in_service_ts) != 0 else math.inf for x in shift])

            if ta == min(min_agent_ts,ta):
                #add another individual
                Na = Na+1
                t=ta
                ind = Individuals(t,Na)
                self.queue.append(ind)
                self.history.append(ind)
                if int(ta/60) < 5 or int(1080 - ta/60) < 5:
                    ta = rnd.expovariate(self.demand[5])

                if len(agents) != 0:
                    #amend the system time
                    ind_to_serve = self.queue.pop(0)
                    shift[agents[0][1]].add_to_queue(ind_to_serve,t)
                else:
                    ind = self.queue[-1]
                    ind.wait_time = min_agent_ts - t 
            else:
                
                #ammend the system time
                t = min_agent_ts
                shift[agents[0][1]].depart_customer_from_service_line(t)

        return shift, self.history

    def generate_demand(self, work_day_hours: int):
        """
        Generates a workday expectation of volume offered to servers
        Returns a list of demand of discrete time intervals (minutes)
        is a concave quadratic continuous function in the R2 space
        """
        t = np.linspace(0,work_day_hours,work_day_hours*60*60)
        demand = -6*t*(t-work_day_hours)
        return demand

class Servers():

    def __init__(self, mean_ts, id, conc_lim):
        self.arival_times = []
        self.customer_number = []
        self.departure_times = []
        self.server_times = []
        self.server_queue = []
        self.service = [] #this is the operating service and limited by the concurrent chats pair(Individual, departure_time)
        self.idle_time = []
        self.concurrent_chats = 0
        self.in_service_ts = []
        self.mean_service_time = mean_ts
        self.id = id
        self.concurrency_measure = []

    def add_to_queue(self, ind: Individuals, t):
        self.server_queue.append(ind)
        
        if self.concurrent_chats < 3:
            self.add_to_service_line(ind, t)
            
    def add_to_service_line(self, ind: Individuals, t):
        self.server_queue.pop() 
        self.concurrent_chats = self.concurrent_chats + 1
        self.customer_number.append(ind.id)
        #generate a service time
        ts = rnd.expovariate(1/self.mean_service_time)*60.
        self.in_service_ts.append(ts)
        self.server_times.append(ts)
        self.departure_times.append(ts+t)
        ind.departure_time = ts+t
        self.service.append((ind, ts+t))
        self.concurrency_measure.append((self.concurrent_chats, t))

    def min_service_time(self):
        return min(self.in_service_ts)

    def depart_customer_from_service_line(self,t):
        self.service = [(ind, td) for ind,td in self.service if td > t]
        self.concurrent_chats = self.concurrent_chats - 1
        self.in_service_ts.remove[t]




        
        
        
        