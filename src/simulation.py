import numpy as np
import tqdm
from src.reception import Reception
from src.room import Room
from src.action import Action, ACTION_ARRIVAL
import heapq
from collections import defaultdict


class Simulation:
    def __init__(self, simulation_count, hparams):
        self.simulation_count = simulation_count
        self.max_patient_wait = hparams['max_patient_wait']
        self.arrival_rate = hparams['arrival_rate']
        self.reception = Reception(
            self, [Room(self, i + 1, mu) for i, mu in enumerate(hparams['rooms'])], hparams['reception_mu'])

        self.bored_patients = set()
        self.finished_patients = set()
        self.bored_patients_count = np.zeros(2, np.int32)
        self.num_patients = 0
        self.time = np.zeros(1, np.double)
        self.action_list = []
        self.progress = None
        self.if_check_accuracy = False

        # results
        self.simulated_count = np.zeros(2, np.int32)
        self.time_spent_history = [[],[]]
        self.accuracy_history = [0]
        self.attendance_history = []

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
        self.progress = tqdm.tqdm(total=self.simulation_count)
        # self.shit = defaultdict(list)
        # self.done = list()
        self.add_action(Action(0, 0, ACTION_ARRIVAL))
        # try:
        while self.action_list:
            action = heapq.heappop(self.action_list)
            # print("action:", action)
            action.execute(self)
            del action
            # print("list:", sorted(self.action_list))

            # self.shit[action.id].append(action)
            # print("p history:", self.shit[action.id])
            # self.done.append(action)
            # print("done:", sorted(self.done)[::-1])
            # print(f'q1({self.reception.q1_len}):', self.reception.q1, f'q1({self.reception.q0_len}):',
            #       self.reception.q0, )
            # input()
            # print("action:", action)
        # except :
        #     print("action:", action)
        #     print("p history:", self.shit[action.id])
        self.progress.close()
    def check_if_queue(self):
        if self.reception.if_queue:
            return False
        else:
            for room in self.reception.rooms:
                if room.if_queue:
                    return False
        return True

    def check_accuracy(self):
        if not self.if_check_accuracy:
            f = np.array(self.time_spent_history[0]+self.time_spent_history[1]).flatten()
            sd = np.std(f)
            mean = np.mean(f)
            acc = 1.96 * sd / (np.sqrt(len(f)) * mean)
            if 1 - acc > 0.95 and acc != 0:
                print("Needed number of patients for 95% of accuracy: ", len(f))
                self.if_check_accuracy = True
        return 0.
