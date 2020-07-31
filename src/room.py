import numpy as np
from src.queue import QueMixin
from src.patient import Patient


class Room(QueMixin):
    def __init__(self, simulation, number, doc_mu: np.array):
        super().__init__(simulation, number)
        self.doc_mu = doc_mu
        self.doc_available = np.array([True for i in range(len(doc_mu))], dtype=bool)
        self._doc_numbers = np.arange(len(doc_mu))

    def assign_doctor(self, patient: Patient):
        doc_args = self._doc_numbers[self.doc_available]
        if len(doc_args):
            doc = np.random.choice(doc_args, 1)
            patient.dr_id = doc
            self.doc_available[doc] = False
            patient.queue = None
            return True
        return False

    def get_service_time(self, doc):
        return np.random.exponential(self.doc_mu[doc], 1)

    def departure(self, patient):
        self.doc_available[patient.dr_id] = True
        self.register_finished_service(patient, wait_time=patient.start - patient.scheduler_departure, )
