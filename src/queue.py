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
            self.simulation.time_spent_history.append(
                patient.checkpoint - patient.scheduler_arrival + self.simulation.max_patient_wait)
            self.simulation.finished_patients.add(patient.id)
        if self.id:
            self.simulation.time_spent_history.append(patient.departure - patient.scheduler_arrival)
            self.simulation.finished_patients.add(patient.id)

    def register_bored_patient(self, patient):
        patient.bored = True
        patient.departure = patient.checkpoint + self.simulation.max_patient_wait
        self.bored_patients_count[patient.type] += 1
        self.register_finished_service(patient, self.simulation.max_patient_wait)

    def mean_length(self):
        t = np.array([self.len_history_timestamps[i] - self.len_history_timestamps[i - 1] for i in
                      range(1, len(self.len_history_timestamps))])
        q0 = np.array(self.len_history[0][:len(t) - 1]) * t
        q1 = np.array(self.len_history[1][:len(t) - 1]) * t
        if self.len_history_timestamps:
            last_time = self.len_history_timestamps[-1]
            return (q0 + q1).sum() / last_time, q0.sum() / last_time, q1.sum() / last_time
        return 0., 0., 0.

    def __eq__(self, other):
        return other and self.id == other.id

    def plot_room_history(self):
        plt.plot(self.len_history_timestamps, self.len_history[1], label='Corona Positive Patients')
        plt.plot(self.len_history_timestamps, self.len_history[0], label='Corona Negative Patients')
        plt.title('Que #{} - Length/Time'.format(self.id))
        plt.legend()
        plt.show()

    def __repr__(self):
        return 'Que #{} - time:{}\n\t* len: {} - ({}, {})\n\t* mean len:{} - ({}, {})\n\tmean wait:{} - ({}, {})\n'.format(
            self.id, self.last_checkpoint, len(self), self.q0_len, self.q1_len, *self.mean_length(),
            0 if self.simulated_count.sum() == 0 else np.array(self.wait_history[0] + self.wait_history[1]).mean(),
            0 if self.simulated_count[0] else np.array(self.wait_history[0]).mean(),
            0 if self.simulated_count[1] else np.array(self.wait_history[1]).mean()
        )
