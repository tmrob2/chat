import math
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import deque
import sys
from scipy import stats

class Individuals():
    def __init__(self, arrival_time, id):
        self.arrival_time = arrival_time
        self.id = id
        self.wait_time = 0.
        self.departure_time = 0.
        self.abandoned = 0
        self.enquiry = ""


class SimulationModel():
    def __init__(self, service_rate, server_count):
        self.queue = deque()
        self.demand = self.generate_demand(18)
        self.history = {}
        self.service_rate = service_rate
        self.server_count = server_count
        t = np.linspace(1, 10000, 10000)
        self.survival = np.exp(-np.exp(-7.5 / 0.902) * t ** (1 / 0.902))
        self.indiviuals_in_service = []

    def MMSNPS_simulation_loop_singlesim(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):

        s_id = 0
        t, na, c1, c2 = 0, 0, 0, 0
        t0 = rnd.expovariate(self.demand[5]) * 60.
        ta = t0
        id_active = 0
        # generate a two server model
        dt = pd.read_csv("overallshift.csv")
        day_of_week = 'Sat'
        schedule = dt.query('start <= %s < end' % t0)[day_of_week].values[0]

        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(schedule)]
        s_id = 46
        # generate the first service time

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            if dt.query('start <= %s < end' % t)[day_of_week].values[0] - 1 - len(shift) >= 0:
                diff = dt.query('start <= %s < end' % t)[day_of_week].values[0] - 1 - len(shift)
                for i in range(0, diff):
                    shift.append(Servers(self.service_rate, s_id, concurency_limit))
                    s_id += 1
            elif dt.query('start <= %s < end' % t)[day_of_week].values[0] - len(shift) + id_active < 0:
                diff = dt.query('start <= %s < end' % t)[day_of_week].values[0] - len(shift) + id_active
                for i in range(int(math.fabs(diff))):
                    shift[id_active].active = False
                    id_active += 1

            applicable_shift = [agent for agent in shift if agent.active is True]
            applicable_shift.sort(key=lambda x: (x.min_next_service, -x.concurrency))
            min_agent_ts = min([agent.min_td for agent in shift])

            if ta == min(min_agent_ts, ta):
                # The generated time interval is less than the next service time

                # Compute the agent idle time
                self.compute_idle_time(applicable_shift, ta, t)
                # Add another individual
                na += 1
                t = ta
                ind = Individuals(t, na)
                self.history[ind.id] = ind

                if int(ta / 60) < 5 or int(1080 - ta / 60) < 5:
                    ta = rnd.expovariate(self.demand[5]) * 3600 + t
                else:
                    ta = rnd.expovariate(self.demand[int(t / 60)]) * 3600 + t

                applicable_shift[0].add_to_queue(ind, t, self.survival)

            else: 
                # The generated time interval is greater than the generated time
                t = min_agent_ts
                # an agent in the shift will process the person in the service line
                [agent.depart_from_service_line(t, self.survival) for agent in shift if agent.min_td == t]

        min_queue = min([len(agent.queue) for agent in shift])
        while min_queue > 0:
            min_td = [agent.min_td for agent in shift]

            #Compute the agent idle time
            self.compute_idle_time(shift, min_td, t)

            t = min_td
            agents_with_qs = [agent for agent in shift if len(agent.queue) > 0]

            for agent in agents_with_qs:
                agent.depart_from_service_line(t, self.survival)

            min_queue = min([len(agent.queue) for agent in shift])

        return shift, self.history

    def MMSNPS_simulation_loop_singlesim_seg_qs(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):

        s_id = 0
        t, na, c1, c2 = 0, 0, 0, 0
        t0 = rnd.expovariate(self.demand[5]) * 60.
        ta = t0
        id_active = 0
        # generate a two server model
        day_of_week = 'Tue'
        dt = pd.read_csv("overallshift.csv")
        schedule = dt.query('start <= %s < end' % t0)[day_of_week].values[0]
        shift_changes = deque([3600, 7200, 18000, 21600, 25200, 32400, 36000, 39600, 50400, 54000, 57600])
        shift = []
        #Create the roster schedules
        working_roster = Shift()
        tmo_first_shift = working_roster.TMO.query('start <= %s < end' % t0)[day_of_week].values[0]
        macomp_first_shift = working_roster.MACOMP.query('start <= %s < end' % t0)[day_of_week].values[0]
        maenq_first_shift = working_roster.MAENQ.query('start <= %s < end' % t0)[day_of_week].values[0]
        mobile_first_shift = working_roster.MOBILE.query('start <= %s < end' % t0)[day_of_week].values[0]
        repair_first_shift = working_roster.REPAIR.query('start <= %s < end' % t0)[day_of_week].values[0]
        tvsport_first_shift = working_roster.TVSPORT.query('start <= %s < end' % t0)[day_of_week].values[0]

        for i in range(tmo_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'tmo'))

        for i in range(macomp_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'macomp'))

        for i in range(maenq_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'maenq'))

        for i in range(mobile_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'mob'))

        for i in range(repair_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'rep'))

        for i in range(tvsport_first_shift):
            shift.append(Servers(self.service_rate, i, concurency_limit, worker_type = 'tvsport'))
        
        # generate the first service time
        s_id = tmo_first_shift+macomp_first_shift+maenq_first_shift+mobile_first_shift+repair_first_shift+tvsport_first_shift
        
        tmo_offline = 0
        while t < max_sim_time:
            
            if t < 32400 and self.shift_change(t, shift_changes) is True:
                tmo_first_shift = working_roster.TMO.query('start <= %s < end' % t)[day_of_week].values[0]
                macomp_first_shift = working_roster.MACOMP.query('start <= %s < end' % t)[day_of_week].values[0]
                maenq_first_shift = working_roster.MAENQ.query('start <= %s < end' % t)[day_of_week].values[0]
                mobile_first_shift = working_roster.MOBILE.query('start <= %s < end' % t)[day_of_week].values[0]
                repair_first_shift = working_roster.REPAIR.query('start <= %s < end' % t)[day_of_week].values[0]
                tvsport_first_shift = working_roster.TVSPORT.query('start <= %s < end' % t)[day_of_week].values[0]

                for i in range(tmo_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'tmo'))

                for i in range(macomp_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'macomp'))

                for i in range(maenq_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'maenq'))

                for i in range(mobile_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'mob'))

                for i in range(repair_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'rep'))

                for i in range(tvsport_first_shift):
                    s_id += 1
                    shift.append(Servers(self.service_rate, s_id, concurency_limit, worker_type = 'tvsport'))

            elif t >= 32400 and self.shift_change(t, shift_changes) is True:
                tmo_delta = working_roster.TMO.query('start <= %s < end' % t)[day_of_week].values[0]
                macomp_delta = working_roster.MACOMP.query('start <= %s < end' % t)[day_of_week].values[0]
                maenq_delta = working_roster.MAENQ.query('start <= %s < end' % t)[day_of_week].values[0]
                mobile_delta = working_roster.MOBILE.query('start <= %s < end' % t)[day_of_week].values[0]
                repair_delta = working_roster.REPAIR.query('start <= %s < end' % t)[day_of_week].values[0]
                tvsport_delta = working_roster.TVSPORT.query('start <= %s < end' % t)[day_of_week].values[0]

                tmo = [i for i in shift if i.worker_type == 'tmo' if i.active is True]
                count = 0
                while count < math.fabs(tmo_delta):
                    tmo[count].active = False
                    count += 1

                macomp = [i for i in shift if i.worker_type == 'macomp' if i.active is True]
                count = 0
                while count < math.fabs(macomp_delta):
                    macomp[count].active = False
                    count += 1

                maenq = [i for i in shift if i.worker_type == 'maenq' if i.active is True]
                count = 0
                while count < math.fabs(maenq_delta):
                    maenq[count].active = False
                    count += 1

                mobile = [i for i in shift if i.worker_type == 'mob' if i.active is True]
                count = 0
                while count < math.fabs(mobile_delta):
                    mobile[count].active = False
                    count += 1

                repair = [i for i in shift if i.worker_type == 'rep' if i.active is True]
                count = 0
                while count < math.fabs(repair_delta):
                    repair[count].active = False
                    count += 1

                tvsport = [i for i in shift if i.worker_type == 'tvsport' if i.active is True]
                count = 0
                while count < math.fabs(tvsport_delta):
                    tvsport[count].active = False
                    count += 1
            
            # Generate the problems here
            p = [0., 0.354, 0.654, 0.797, 0.897, 0.976, 1.]

            if t < 7200 or t > 50400:
                r = rnd.uniform(0, 0.897)

            l = ['rep', 'maenq', 'tvsport', 'mob', 'tmo', 'macomp']
            w_type = l[self.generate_random_sample_index(r, p)]
            
            applicable_shift = [agent for agent in shift if agent.active is True if agent.worker_type == w_type]
            applicable_shift.sort(key=lambda x: (x.min_next_service, -x.concurrency))
            min_agent_ts = min([agent.min_td for agent in shift])

            if ta == min(min_agent_ts, ta):
                # The generated time interval is less than the next service time
                
                # Compute the agent idle time
                self.compute_idle_time(applicable_shift, ta, t)
                # Add another individual
                na += 1
                t = ta
                ind = Individuals(t, na)
                # Generate the individuals problem
                self.history[ind.id] = ind
                ind.enquiry = w_type

                if int(ta / 60) < 5 or int(1080 - ta / 60) < 5:
                    ta = rnd.expovariate(self.demand[5]) * 3600 + t
                else:
                    ta = rnd.expovariate(self.demand[int(t / 60)]) * 3600 + t

                applicable_shift[0].add_to_queue(ind, t, self.survival)

            else: 
                # The generated time interval is greater than the generated time
                t = min_agent_ts
                # an agent in the shift will process the person in the service line
                [agent.depart_from_service_line(t, self.survival) for agent in shift if agent.min_td == t]

        min_queue = min([len(agent.queue) for agent in shift])
        while min_queue > 0:
            min_td = [agent.min_td for agent in shift]

            #Compute the agent idle time
            self.compute_idle_time(shift, min_td, t)

            t = min_td
            agents_with_qs = [agent for agent in shift if len(agent.queue) > 0]

            for agent in agents_with_qs:
                agent.depart_from_service_line(t, self.survival)

            min_queue = min([len(agent.queue) for agent in shift])

        return shift, self.history
    
    def MMSNPS_simulation_loop_multisim(self, max_sim_time, concurency_limit: int, model_type = 'mbf'):
        s_id = 0
        t, na, c1, c2 = 0, 0, 0, 0
        t0 = rnd.expovariate(self.demand[5]) * 60.
        ta = t0
        id_active = 0
        # generate a two server model
        day_of_week = 'Mon'
        dt = pd.read_csv("overallshift.csv")
        schedule = dt.query('start <= %s < end' % t0)[day_of_week].values[0]

        shift = [Servers(self.service_rate, i, concurency_limit) for i in range(schedule)]
        s_id = 46
        # generate the first service time

        while t < max_sim_time:
            # loop through the servers to see who is the most busy but not reached their concurrency limit
            if dt.query('start <= %s < end' % t)[day_of_week].values[0] - 1 - len(shift) >= 0:
                diff = dt.query('start <= %s < end' % t)[day_of_week].values[0] - 1 - len(shift)
                for i in range(0, diff):
                    shift.append(Servers(self.service_rate, s_id, concurency_limit))
                    s_id += 1
            elif dt.query('start <= %s < end' % t)[day_of_week].values[0] - len(shift) + id_active < 0:
                diff = dt.query('start <= %s < end' % t)[day_of_week].values[0] - len(shift) + id_active
                for i in range(int(math.fabs(diff))):
                    shift[id_active].active = False
                    id_active += 1

            applicable_shift = [agent for agent in shift if agent.active is True]
            applicable_shift.sort(key=lambda x: (x.min_next_service, -x.concurrency))
            min_agent_ts = min([agent.min_td for agent in shift])

            if ta == min(min_agent_ts, ta):
                # The generated time interval is less than the next service time

                # Compute the agent idle time
                self.compute_idle_time(applicable_shift, ta, t)
                # Add another individual
                na += 1
                t = ta
                ind = Individuals(t, na)
                self.history[ind.id] = ind

                if int(ta / 60) < 5 or int(1080 - ta / 60) < 5:
                    ta = rnd.expovariate(self.demand[5]) * 3600 + t
                else:
                    ta = rnd.expovariate(self.demand[int(t / 60)]) * 3600 + t

                applicable_shift[0].add_to_queue(ind, t, self.survival)

            else: 
                # The generated time interval is greater than the generated time
                t = min_agent_ts
                # an agent in the shift will process the person in the service line
                [agent.depart_from_service_line(t, self.survival) for agent in shift if agent.min_td == t]

        min_queue = min([len(agent.queue) for agent in shift])
        while min_queue > 0:
            min_td = [agent.min_td for agent in shift]

            #Compute the agent idle time
            self.compute_idle_time(shift, min_td, t)

            t = min_td
            agents_with_qs = [agent for agent in shift if len(agent.queue) > 0]

            for agent in agents_with_qs:
                agent.depart_from_service_line(t, self.survival)

            min_queue = min([len(agent.queue) for agent in shift])

        avg_wait_time = np.average([self.history[i].wait_time for i in range(1,len(self.history))if self.history[i].wait_time != 0.]) 
        median_wait_time = np.median([self.history[i].wait_time for i in range(1,len(self.history)) if self.history[i].wait_time != 0.])
        max_wait_time = np.max([self.history[i].wait_time for i in range(1,len(self.history))])
        avg_service_time = np.average([np.average(i.service_times_history) for i in shift])
        avg_ar = np.average([self.history[i].abandoned for i in range(1,len(self.history))])
        sum_chats_answered = sum([1 for i in range(1,len(self.history)) if self.history[i].abandoned == 0 if self.history[i].departure_time > 0])
        return avg_wait_time, median_wait_time, max_wait_time, avg_service_time, avg_ar, sum_chats_answered

    def compute_idle_time(self, shift, t1, t0):
        for agent in shift:
            if agent.concurrency == 0 and agent.active is True:
                agent.idle += (t1-t0)
                agent.idle_time.append(t1)

    def generate_demand(self, work_day_hours: int):
        """
        Generates a workday expectation of volume offered to servers
        Returns a list of demand of discrete time intervals (minutes)
        is a concave quadratic continuous function in the R2 space
        """
        t = np.linspace(0, work_day_hours, work_day_hours * 60)
        demand = -10 * t * (t - work_day_hours)
        return demand

    def plot_wait_time(self):
        wt = []
        for i in range(1,len(self.history)):
            if self.history[i].wait_time != 0:
                wt.append(self.history[i].wait_time)
        plt.figure()
        plt.hist(wt, 100)
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
        plt.plot(x[:-1], n*bins)
        plt.title('Count of agents idle at time t')
        plt.savefig("staff_idle_hours.png")
        return n, x

    def shift_change(self, t, times):
        new_shift_times = times
        if t > new_shift_times[0]:
            new_shift_times.popleft()
            return True
        else:
            return False

    def generate_random_sample_index(self, r, p):
        for i in range(0,len(p)):
            if (p[i] <= r) and (r <= p[i+1]): 
                return i

class Servers:

    def __init__(self, mean_ts, id, conc_lim, worker_type = ""):
        self.service_times_history = []
        self.concurrency = 0
        self.c_lim = conc_lim
        self.queue = deque()
        self.service_line = deque()
        self.active = True
        self.mean_ts = mean_ts
        self.min_td = math.inf
        self.min_next_service = 0
        self.idle = 0.
        self.idle_time = []
        self.conc_measure = 0.
        self.worker_type = worker_type

    def add_to_queue(self, ind: Individuals, t, survival):
        # Append a customer to the queue straight away
        self.queue.append(ind)
        # If the agent has capacity the queue will be processed
        while len(self.queue) > 0 and self.concurrency < self.c_lim:
            customer = self.queue.popleft()
            t_service = int(rnd.expovariate(1/self.mean_ts))*60
            self.service_times_history.append(t_service)
            customer.departure_time = t + t_service
            self.conc_measure += customer.departure_time - customer.arrival_time
            self.service_line.append(customer)
            self.concurrency += 1

        self.get_min_departure_time()
        ls_ab = self.calculate_wait_time(t, survival)

        for cust in ls_ab:
            self.queue.remove(cust)

    def calculate_wait_time(self, t, survival):
        ls_abandoned = []
        for i in self.queue:
            i.wait_time = t - i.arrival_time
            if self.generate_abandoment(i.wait_time, survival) is True:
                ls_abandoned.append(i)
                i.abandoned = 1
        return ls_abandoned

    def get_min_departure_time(self):
        min_time = math.inf
        next_service = 0
        for i in self.service_line:
            if i.departure_time < min_time:
                min_time = i.departure_time

            if i.departure_time > next_service:
                next_service = i.departure_time
        self.min_td = min_time
        self.min_next_service = next_service

    def depart_from_service_line(self, t, survival):
        ls_remove = []
        ls_append = []
        for i in self.service_line:
            if i.departure_time == t:
                ls_remove.append(i)
                self.concurrency -= 1
        while self.queue.__len__() > 0 and self.concurrency < self.c_lim:
            customer = self.queue.popleft()
            customer.departure_time = t + int(rnd.expovariate(1/self.mean_ts))*60
            ls_append.append(customer)
        [self.service_line.remove(i) for i in ls_remove]
        for i in ls_append:
            self.concurrency += 1
            self.service_line.append(i)
        self.get_min_departure_time()
        ls_ab = self.calculate_wait_time(t, survival)

        for cust in ls_ab:
            self.queue.remove(cust)

    def generate_idle_flag(self, t):
        if self.concurrency == 0:
            self.idle_time.append(int(t/60))

    def generate_abandoment(self, wait_time, survival):
        p = rnd.uniform(0,1)
        if p > survival[int(wait_time)]:
            return True
        else:
            return False

class Shift:

    def __init__(self):
        self.TMO = pd.read_csv("TMOshift.csv")
        self.MACOMP = pd.read_csv("macompshift.csv")
        self.MAENQ = pd.read_csv("maenqshift.csv")
        self.MOBILE = pd.read_csv("mobileshift.csv")
        self.REPAIR = pd.read_csv("repairshift.csv")
        self.TVSPORT = pd.read_csv("tvsportshift.csv")
        self.overall = pd.read_csv("overallshift.csv")

     