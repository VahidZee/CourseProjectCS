import numpy as np


class Patient:
    def __init__(self, patient_id, scheduler_arrival):
        self.id = patient_id
        self.scheduler_arrival = scheduler_arrival
        self.checkpoint = scheduler_arrival
        self.scheduler_start = None
        self.scheduler_departure = None
        self.start = None
        self.departure = None
        self._set_type()
        self.bored = False
        self.room = None
        self.queue = None
        self.dr_id = None

    def _set_type(self):
        self.type = np.random.binomial(1, 0.1, 1)[0]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return "#{} - q:{} r:{}\n* {} - {} {} - {} {}\n".format(
            self.id, -1 if self.queue is None else self.queue.id, 0 if self.room is None else self.room.id,
            self.scheduler_arrival, self.scheduler_start, self.scheduler_departure,
            self.start, self.departure
        )
