from src.queue import QueMixin
from src.patient import Patient
import numpy as np


class Reception(QueMixin):
    def __init__(self, simulation, rooms, mu):
        super().__init__(simulation, 0)
        self.rooms = rooms
        self.rooms_len = np.zeros(len(rooms))
        self.mu = mu
        self.rooms_q_len = np.zeros(len(rooms))

    def get_service_time(self):
        return np.random.exponential(self.mu)

    def find_room(self):
        min_room_q_len = np.min(self.rooms_q_len)
        return self.rooms[np.random.choice(np.where(self.rooms_q_len == min_room_q_len)[0], 1)[0]]

    def assign_room(self, patient: Patient):
        room = self.find_room()
        patient.queue = room
        patient.room = room
        self.register_finished_service(patient, patient.scheduler_start - patient.scheduler_arrival)
        patient.checkpoint = patient.scheduler_departure
        self.remove_head(patient, patient.checkpoint)
        return room
