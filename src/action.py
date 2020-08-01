from src.patient import Patient
from src.queue import QueMixin
from src.util import Node
import copy

ACTION_ARRIVAL = 1
ACTION_RECEPTION_DEPARTURE = 2
ACTION_VISIT_START = 3
ACTION_VISIT_DEPARTURE = 4
ACTION_BORED = 5


class Action:
    def __init__(self, id, time, action_type, patient: Patient = None, queue: QueMixin = None, node: Node = None,
                 # par=None
                 ):
        self.time = time
        self.type = action_type
        self.patient = patient
        self.id = id
        self.node = node
        self.queue = queue
        # self.parent = par

    def _arrrival_exec(self, simulation):
        self.patient = Patient(copy.copy(simulation.num_patients), self.time)
        node = simulation.reception.add_patient(self.patient, self.time)
        simulation.num_patients += 1
        simulation.register_attendance(self.patient.type, +1, self.time)
        simulation.progress.update(1)
        if len(simulation.reception) == 1:
            self.patient.scheduler_start = self.time
            self.patient.scheduler_departure = self.time + simulation.reception.get_service_time()
            simulation.add_action(
                Action(id=self.id, time=self.patient.scheduler_departure, action_type=ACTION_RECEPTION_DEPARTURE,
                       patient=self.patient, node=node, ))
        else:
            simulation.add_action(
                Action(id=self.id, time=self.time + simulation.max_patient_wait, action_type=ACTION_BORED,
                       patient=self.patient, queue=simulation.reception, node=node, ))
        if simulation.num_patients < simulation.simulation_count:
            simulation.add_action(
                Action(id=simulation.num_patients, time=self.time + simulation.get_arrival_interval(),
                       action_type=ACTION_ARRIVAL, ))

    def _reception_departure_exec(self, simulation):
        if self.id in simulation.bored_patients:
            return
        # print('assign room', self.patient.id)
        room = simulation.reception.assign_room(
            self.patient)  # changes patient queue and checkpoint assigns it to the best room
        node = room.add_patient(self.patient, self.time)
        if len(simulation.reception):
            next_node = simulation.reception.head()
            # print(next_node, simulation.reception.q0_len, simulation.reception.q1_len, len(simulation.reception),
            #       simulation.reception.q0, simulation.reception.q1)
            next_patient = next_node.patient
            next_patient.scheduler_start = self.time
            next_patient.scheduler_departure = self.time + simulation.reception.get_service_time()
            simulation.add_action(
                Action(id=next_patient.id, time=next_patient.scheduler_departure,
                       action_type=ACTION_RECEPTION_DEPARTURE, patient=next_patient, node=next_node, ))

        if room.assign_doctor(self.patient):
            self.patient.start = self.time
            simulation.add_action(
                Action(id=self.id, time=self.time, action_type=ACTION_VISIT_START, patient=self.patient, node=node, ))
        else:
            simulation.add_action(
                Action(self.id, time=self.time + simulation.max_patient_wait, action_type=ACTION_BORED,
                       patient=self.patient, queue=room, node=node, ))

    def _visit_start_exec(self, simulation):
        if self.id in simulation.bored_patients:
            return
        self.patient.room.remove_node(self.node, self.time)
        self.patient.departure = self.patient.start + self.patient.room.get_service_time(self.patient.dr_id)
        simulation.add_action(
            Action(id=self.id, time=self.patient.departure, action_type=ACTION_VISIT_DEPARTURE,
                   patient=self.patient, node=self.node, ))

    def _visit_departure_exec(self, simulation):
        self.patient.room.departure(self.patient)
        simulation.register_attendance(self.patient.type, -1, self.time)
        # simulation.accuracy_history.append(simulation.check_accuracy()) # Todo: check?
        if len(self.patient.room):
            next_node = self.patient.room.head()
            next_patient = next_node.patient
            next_patient.start = self.time
            simulation.add_action(
                Action(id=next_patient.id, time=self.time, action_type=ACTION_VISIT_START, patient=next_patient,
                       node=next_node, ))

    def _bored_exec(self, simulation):
        if self.id not in simulation.finished_patients and self.queue == self.patient.queue:
            # print(self.id, "got bored!!")
            # print("----------")
            # if self.time != self.patient.checkpoint + simulation.max_patient_wait:
            #     print('well fuq')
            # print(self.queue.q0_len, self.queue.q1_len)
            simulation.register_attendance(self.patient.type, -1, self.time)
            self.queue.remove_node(self.node, self.patient.checkpoint + simulation.max_patient_wait)
            # print(self.queue.q0_len, self.queue.q1_len)

            simulation.bored_patients_count[self.patient.type] += 1
            self.queue.register_bored_patient(self.patient)
            # print(self.queue.q0_len, self.queue.q1_len)

            simulation.bored_patients.add(self.patient.id)
            # print(self.queue.q0_len, self.queue.q1_len)

    def execute(self, simulation):
        if self.type == ACTION_BORED:
            self._bored_exec(simulation)
            return
        if self.type == ACTION_ARRIVAL:
            self._arrrival_exec(simulation)
            return
        if self.type == ACTION_RECEPTION_DEPARTURE:
            self._reception_departure_exec(simulation)
            return
        if self.type == ACTION_VISIT_START:
            self._visit_start_exec(simulation)
            return
        if self.type == ACTION_VISIT_DEPARTURE:
            self._visit_departure_exec(simulation)
            return

    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time

    def __eq__(self, other):
        return self.time == other.time

    def __ge__(self, other):
        return self.time >= other.time

    def __gt__(self, other):
        return self.time > other.time

    def __repr__(self):
        action = ""
        if self.type == ACTION_BORED:
            action = "bored"
        if self.type == ACTION_ARRIVAL:
            action = "arrival"
        if self.type == ACTION_RECEPTION_DEPARTURE:
            action = "reception departure"
        if self.type == ACTION_VISIT_START:
            action = "visit start"
        if self.type == ACTION_VISIT_DEPARTURE:
            action = "visit departure"

        # par_action = ""
        # if self.parent is not None and self.parent.type == ACTION_BORED:
        #     par_action = "bored"
        # if self.parent is not None and self.parent.type == ACTION_ARRIVAL:
        #     par_action = "arrival"
        # if self.parent is not None and self.parent.type == ACTION_RECEPTION_DEPARTURE:
        #     par_action = "reception departure"
        # if self.parent is not None and self.parent.type == ACTION_VISIT_START:
        #     par_action = "visit start"
        # if self.parent is not None and self.parent.type == ACTION_VISIT_DEPARTURE:
        #     par_action = "visit departure"
        # return "{:03}-{}-{}({})".format(self.id, self.time, action, '' if self.parent is None else
        # '{},{},{}'.format(self.parent.time, self.parent.id, par_action))
        return "{:03}-{}-{}".format(self.id, self.time, action)
