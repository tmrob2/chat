# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:13:50 2017

@author: 611004435
"""

import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Individuals():
    
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time
        
    def enter_queue(self, previous_exit_time, service_rate):
        self.service_time = rnd.expovariate(service_rate)
        self.service_start = max(self.arrival_time, previous_exit_time)
        self.exit_time = self.service_start + self.service_time
        
        
class SimulationModel():
    
    def __init__(self, demand, service_rate):
        self.queue = []
        self.demand = demand
        self.service_rate = service_rate
        
        
    def clean_up_queue(self, t):
        self.queue = [ind for ind in self.queue if ind.exit_time > t]
        
    def main_simulation_loop(self, max_sim_time):
        t = 0
        previous_exit_time = 0
        while t < t_max_sim_time:
            #record the stats of the loops. For example at time t
            #how many people were in the queue
            #how many servers are there
            #what is the service rate
            #what is the abandoment rate
            self.demand = -1.8/1080*t*(t-1080)
            t += rnd.expovariate(self.demand)
            new_individual = Individual(t)
            new_individual.enter_queue(previous_exit_time, self.service_rate)
            self.clean_up_queue(t)
            previous_exit_time = new_individual.exit_time
        
        
        
        
        