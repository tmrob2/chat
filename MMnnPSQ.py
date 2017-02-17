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
        t = np.linspace(1, 10000, 10000)
        self.survival = np.exp(-np.exp(-9.5 / 0.902) * t ** (1 / 0.902))
        self.indiviuals_in_service = []

    def MMS1PS_simulation_loop_singlesim(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):

        s_id = 0
        t, na, c1, c2 = 0, 0, 0, 0
        t0 = rnd.expovariate(self.demand[5]) * 60.
        ta = t0
        id_active = 0
        # generate a two server model
        dt = pd.read_csv("overallshift.csv")
        schedule = dt.query('start <= %s < end' % t0)['Mon'].values[0]

        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(schedule)]
        s_id = 46
        # generate the first service time

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            if dt.query('start <= %s < end' % t)['Mon'].values[0] - 1 - len(shift) >= 0:
                diff = dt.query('start <= %s < end' % t)['Mon'].values[0] - 1 - len(shift)
                for i in range(0, diff):
                    shift.append(Servers(self.service_rate, s_id, concurency_limit))
                    s_id += 1
            elif dt.query('start <= %s < end' % t)['Mon'].values[0] - len(shift) + id_active < 0:
                diff = dt.query('start <= %s < end' % t)['Mon'].values[0] - len(shift) + id_active
                for i in range(int(math.fabs(diff))):
                    shift[id_active].active = False
                    id_active += 1

            shift.sort(key=lambda x: (x.concurrency, x.service)

            if ta == min(min_agent_ts, ta):

    def generate_demand(self, work_day_hours: int):
        """
        Generates a workday expectation of volume offered to servers
        Returns a list of demand of discrete time intervals (minutes)
        is a concave quadratic continuous function in the R2 space
        """
        t = np.linspace(0, work_day_hours, work_day_hours * 60)
        demand = -10 * t * (t - work_day_hours)
        return demand

    def generate_abandoment(self, wait_time):
        p = rnd.uniform(0, 1)
        if p > self.survival[int(wait_time)]:
            return True
        else:
            return False

class Servers:

    def __init__(self, mean_ts, id, conc_lim):
        self.t_arrive = []
        self.t_depart = []
        self.service_times_history = []
        self.concurrency = 0
        self.c_lim = conc_lim
        self.queue = deque()
        self.service_line = deque()
        self.active = True
        self.mean_ts = mean_ts
        self.min_td = math.inf

    def add_to_queue(self, ind: Individuals, t):
        if len(self.service_line) < self.c_lim:
            # Check if there is anybody in the queue
            for i in range(self.concurrency, self.c_lim):
                if len(self.queue) > 0:
                    customer = self.queue.popleft()
                    customer.departure_time = t + int(rnd.expovariate(1/self.mean_ts))*60
                    self.service_line.append()
                    self.concurrency += 1
                else:
                    self.service_line.append(ind)
                    self.concurrency += 1
        else:
            self.queue.append(ind)
        self.get_min_departure_time()
        self.calculate_wait_time()

    def calculate_wait_time(self, t):
        for i in self.queue:
            i.wait_time = t - i.arrival_time

    def get_min_departure_time(self):
        min_time = math.inf
        for i in self.queue:
            if i.departure_time < min_time:
                min_time = i.departure_time
        self.min_td = min_time

