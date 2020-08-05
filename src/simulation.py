import numpy as np
import tqdm
from src.reception import Reception
from src.room import Room
from src.action import Action, ACTION_ARRIVAL
import heapq
import matplotlib.pyplot as plt
import copy


class Simulation:
    def __init__(self, simulation_count, hparams, disable_progress=False):
        self.simulation_count = simulation_count
        self.max_patient_wait = hparams['max_patient_wait']
        self.arrival_rate = 1. / hparams['arrival_rate']
        self.reception = Reception(
            self, [Room(self, i + 1, 1. / mu) for i, mu in enumerate(hparams['rooms'])], 1. / hparams['reception_mu'])
        self.hparams = hparams
        self.bored_patients = set()
        self.finished_patients = set()
        self.bored_patients_count = np.zeros(2, np.int32)
        self.num_patients = 0
        self.time = np.zeros(1, np.double)
        self.action_list = []
        self.progress = None
        self.disable_progress = disable_progress
        self.if_check_accuracy = False

        # results
        self.simulated_count = np.zeros(2, np.int32)
        self.bored_patients_count = np.zeros(2, np.int32)
        self.time_spent_history = [[], []]
        self.wait_history = [[], []]
        self.accuracy_history = [0]
        self.attendance_history = [[], []]
        self.attendance_history_time_stamps = []

    def get_arrival_interval(self):
        return np.random.exponential(self.arrival_rate)

    def add_action(self, action):
        if action.id not in self.finished_patients:
            heapq.heappush(self.action_list, action)

    @staticmethod
    def get_input_dict():
        m, l, a, mu = input().split()
        hparams = dict()
        hparams['max_patient_wait'] = np.array([a], np.double)
        hparams['arrival_rate'] = np.array([l], np.double)
        hparams['reception_mu'] = np.array([mu], np.double)
        hparams['rooms'] = [np.array(input().split(), dtype=np.double) for i in range(int(m))]
        return hparams

    def simulate(self):
        self.progress = tqdm.tqdm(total=self.simulation_count, disable=self.disable_progress)
        self.add_action(Action(0, 0, ACTION_ARRIVAL))
        while self.action_list:
            action = heapq.heappop(self.action_list)
            self.time = action.time
            action.execute(self)
            del action
        self.progress.close()

    def check_if_queue(self):
        if self.reception.if_queue:
            return False
        else:
            for room in self.reception.rooms:
                if room.if_queue:
                    return False
        return True

    def plot_queue_lens_history(self):
        self.reception.plot_len_history()
        for room in self.reception.rooms:
            room.plot_len_room_history()

    def plot_wait_frequency(self):
        if len(self.wait_history[0]):
            plt.hist(np.array(self.wait_history[0], dtype=np.double).reshape(-1), label='Corona Positive Patients',
                     density=True, alpha=0.6, bins=50)
        if len(self.wait_history[1]):
            plt.hist(np.array(self.wait_history[1], dtype=np.double).reshape(-1), label='Corona Negative Patients',
                     density=True, alpha=0.6, bins=50)
        plt.title('Simulation Wait Frequency')
        plt.xlabel('wait duration')
        plt.legend()
        plt.show()

    def plot_response_time_frequency(self):
        if len(self.time_spent_history[0]):
            plt.hist(np.array(self.time_spent_history[0], dtype=np.double).reshape(-1),
                     label='Corona Positive Patients',
                     density=True, alpha=0.6, bins=50)
        if len(self.time_spent_history[1]):
            plt.hist(np.array(self.time_spent_history[1], dtype=np.double).reshape(-1),
                     label='Corona Negative Patients',
                     density=True, alpha=0.6, bins=50)
        plt.title('Simulation Response Time Frequency')
        plt.xlabel('response duration')
        plt.legend()
        plt.show()

    def plot_attendance_frequency(self):
        if len(self.attendance_history[0]):
            plt.hist(np.array(self.attendance_history[0]).reshape(-1),
                     label='Corona Positive Patients', alpha=0.6, bins=50)
        if len(self.attendance_history[1]):
            plt.hist(np.array(self.attendance_history[1]).reshape(-1),
                     label='Corona Negative Patients', alpha=0.6, bins=50)
        plt.title('Simulation Attendance Frequency')
        plt.xlabel('patients count')
        plt.legend()
        plt.show()

    def mean_time_spent(self):
        m0 = np.array(self.time_spent_history[0], dtype=np.double).reshape(-1)
        m1 = np.array(self.time_spent_history[1], dtype=np.double).reshape(-1)
        return np.append(m0, m1).mean(), m0.mean(), m1.mean()

    def mean_wait_time(self):
        m0 = np.array(self.wait_history[0], dtype=np.double).reshape(-1)
        m1 = np.array(self.wait_history[1], dtype=np.double).reshape(-1)
        return np.append(m0, m1).mean(), m0.mean(), m1.mean()

    def print_sub_results(self):
        print("Reception:")
        print(self.reception)
        print("Rooms:")
        for room in self.reception.rooms:
            print(room)

    def register_attendance(self, ptype: int, change, timestamp):
        if len(self.attendance_history_time_stamps) == 0:
            self.attendance_history_time_stamps.append(timestamp)
            self.attendance_history[ptype].append(change)
            self.attendance_history[1 - ptype].append(0)
        elif self.attendance_history_time_stamps[-1] == timestamp:
            self.attendance_history[ptype][-1] += change
        else:
            self.attendance_history[ptype].append(self.attendance_history[ptype][-1] + change)
            self.attendance_history[1 - ptype].append(self.attendance_history[1 - ptype][-1])
            self.attendance_history_time_stamps.append(timestamp)

    def plot_attendance_history(self):
        plt.plot(self.attendance_history_time_stamps, self.attendance_history[1], label='Corona Positive Patients',
                 alpha=0.6)
        plt.plot(self.attendance_history_time_stamps, self.attendance_history[0], label='Corona Negative Patients',
                 alpha=0.6)
        plt.plot(self.attendance_history_time_stamps,
                 np.array(self.attendance_history[0]) + np.array(self.attendance_history[1]), label='Total', alpha=0.4)
        plt.title('Simulation - Attendance/Time')
        plt.legend()
        plt.show()

    def __repr__(self):
        time_spent = '* mean time spent: {:.07} ({:.07}, {:.07})'.format(*self.mean_time_spent())
        wait_mean = '* mean waiting time: {:.04} ({:.04}, {:.04})'.format(*self.mean_wait_time())
        patients = '* patients {} ({}, {}) - bored {} ({}, {})'.format(
            self.simulated_count.sum(), self.simulated_count[0], self.simulated_count[1],
            self.bored_patients_count.sum(), self.bored_patients_count[0], self.bored_patients_count[1]
        )
        return 'Simulation:\n\t{}\n\t{}\n\t{}'.format(time_spent, wait_mean, patients)

    def check_accuracy(self):
        if not self.if_check_accuracy:
            f = np.append(np.array(self.time_spent_history[1]).reshape(-1),
                          np.array(self.time_spent_history[0]).reshape(-1)).flatten()
            sd = np.std(f)
            mean = np.mean(f)
            acc = 1.96 * sd / (np.sqrt(f.shape[0]) * mean)
            if 1 - acc > 0.95 and acc != 0:
                if not self.disable_progress:
                    print("Needed Number of simulations for 95% accuracy is:", f.shape[0])
                self.if_check_accuracy = True
        return 0.

    def find_optimum_service_rate(self, scale=1.5, sim_size=10 ** 4):
        mean_mu, mean_length = [], []
        finished = False
        hparams = copy.deepcopy(self.hparams)
        mu = np.array([room.mean() for room in hparams['rooms']]).mean()
        while not finished:
            hparams['rooms'] = [room * scale for room in hparams['rooms']]
            mu = mu * scale
            sim = Simulation(sim_size, hparams, disable_progress=True)
            sim.simulate()
            len_mu = sim.find_mean_length()
            finished = (len_mu == 0.)
            mean_mu.append(mu)
            mean_length.append(len_mu)
        plt.plot(np.array(mean_mu, dtype=np.double).flatten(), np.array(mean_length, dtype=np.double).flatten())
        plt.xlabel('Average Service Rate')
        plt.ylabel('Average Queue Length')
        plt.title('Optimum Doctors Service rate')
        plt.show()

    def find_mean_length(self):
        return np.array([room.mean_length()[0] for room in self.reception.rooms]).mean()
