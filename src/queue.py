from collections import deque
import numpy as np
from src.patient import Patient
import matplotlib.pyplot as plt
from src.util import Node


class QueMixin:
    def __init__(self, simulation, que_id):
        self.simulation = simulation
        self.id = que_id
        self.q0 = None
        self.q0_last = None
        self.q0_len = 0
        self.q1 = None
        self.q1_last = None
        self.q1_len = 0
        self.if_queue = False

        # simulation requirements
        self.last_checkpoint = np.zeros(1, dtype=np.double)

        # results
        self.wait_history = [[], []]
        self.len_history = [[], []]
        self.len_history_timestamps = []
        self.bored_patients_count = np.zeros(2, dtype=np.int32)
        self.simulated_count = np.zeros(2, dtype=np.int32)

    def remove_node(self, node, timestamp=None):
        if self.id:
            self.simulation.reception.rooms_q_len[self.id - 1] -= 1
        if node.patient.type:
            self.q1_len -= 1
            if node.next is None:
                self.q1_last = node.prev
            if node.prev is None:
                self.q1 = node.next
        else:
            self.q0_len -= 1
            if node.next is None:
                self.q0_last = node.prev
            if node.prev is None:
                self.q0 = node.next
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        del node  # todo check
        if timestamp:
            self.register_len_history(timestamp)

    def remove_head(self, patient, timestamp=None):
        if patient.type:
            self.remove_node(self.q1, timestamp)
        else:
            self.remove_node(self.q0, timestamp)

    def add_patient(self, patient: Patient, timestamp=None):
        #print(self.id,self.q0_len+self.q1_len)
        if self.id:
            if len(self.simulation.reception.rooms[self.id-1].doc_mu) <= self.q0_len+self.q1_len:
                self.if_queue = True
        else:
            if 1 <= self.q0_len+self.q1_len:
                self.if_queue = True
        if self.id:
            self.simulation.reception.rooms_q_len[self.id - 1] += 1
        patient.queue = self
        if patient.type:
            node = Node(patient, None, self.q1_last)
            self.q1_last = node
            self.q1_len += 1
        else:
            node = Node(patient, None, self.q0_last)
            self.q0_last = node
            self.q0_len += 1
        if node.prev is not None:
            node.prev.next = node  # chaining together
        if self.q0 is None:
            self.q0 = self.q0_last
        if self.q1 is None:
            self.q1 = self.q1_last
        if timestamp:
            self.register_len_history(timestamp)
        return node

    def __len__(self):
        return self.q0_len + self.q1_len

    def head(self):
        if self.q1 is not None:
            return self.q1
        if self.q0 is not None:
            return self.q0
        return None

    def register_len_history(self, checkpoint_time):
        if checkpoint_time != self.last_checkpoint or not self.len_history_timestamps:
            self.last_checkpoint = checkpoint_time
            self.len_history[0].append(self.q0_len)
            self.len_history[1].append(self.q1_len)
            self.len_history_timestamps.append(checkpoint_time)
        else:
            self.len_history[0][-1] = self.q0_len
            self.len_history[1][-1] = self.q1_len

    def register_finished_service(self, patient: Patient, wait_time):  # wait history and simulation history
        self.wait_history[patient.type].append(wait_time)
        if patient.bored:
            if patient.type:
                self.simulation.time_spent_history[1].append(
                    patient.checkpoint - patient.scheduler_arrival + self.simulation.max_patient_wait)
            else:
                self.simulation.time_spent_history[0].append(
                    patient.checkpoint - patient.scheduler_arrival + self.simulation.max_patient_wait)
            self.simulation.finished_patients.add(patient.id)
        if self.id:
            if patient.type:
                self.simulation.time_spent_history[1].append(patient.departure - patient.scheduler_arrival)
            else:
                self.simulation.time_spent_history[0].append(patient.departure - patient.scheduler_arrival)
            self.simulation.finished_patients.add(patient.id)
        self.simulation.check_accuracy()

    def register_bored_patient(self, patient):
        patient.bored = True
        patient.departure = patient.checkpoint + self.simulation.max_patient_wait
        self.bored_patients_count[patient.type] += 1
        self.register_finished_service(patient, self.simulation.max_patient_wait)

    def mean_length(self) -> tuple:  # returns mean total, neg, positive queue lengths
        if len(self.len_history_timestamps) == 0:
            return 0., 0., 0.
        t = np.array(self.len_history_timestamps).reshape(-1)
        t[1:] = t[1:] - t[:len(t) - 1]
        avg = (np.array(self.len_history) * t).sum(1) / t.sum()
        return avg.sum(), avg[0], avg[1]

    def mean_wait(self) -> tuple:  # returns mean total, neg, positive queue wait duration
        hist0 = np.array(self.wait_history[0], dtype=np.double).reshape(-1)
        hist1 = np.array(self.wait_history[1], dtype=np.double).reshape(-1)
        return np.append(hist0, hist1).mean(), hist0.mean(), hist1.mean()

    def __eq__(self, other):
        return other and self.id == other.id

    def plot_room_history(self):
        plt.plot(self.len_history_timestamps, self.len_history[1], label='Corona Positive Patients')
        plt.plot(self.len_history_timestamps, self.len_history[0], label='Corona Negative Patients')
        plt.title('Que #{} - Length/Time'.format(self.id))
        plt.legend()
        plt.show()

    def __repr__(self):
        len_res = '* mean-len {:.04}:({:.04}, {:.04})'.format(*self.mean_length())
        wait_res = '* mean-wait {:.04}:({:.04}, {:.04})'.format(*self.mean_wait())
        return 'Que #{} - time:{}\n\t* len: {} - ({}, {})\n\t{}\n\t{}\n'.format(
            self.id, self.last_checkpoint, len(self), self.q0_len, self.q1_len, len_res, wait_res)
