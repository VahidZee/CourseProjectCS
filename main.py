from src.simulation import Simulation

# todo remove
def get_statistics(patients, patients_type_one, patients_type_two):
    time_in_system = 0
    time_in_system_type_one = 0
    time_in_system_type_two = 0
    num_of_bored = 0
    num_of_bored_type_one = 0
    num_of_bored_type_two = 0

    time_in_q = 0
    time_in_q_type_one = 0
    time_in_q_type_two = 0

    time_in_system_list = []
    time_in_system_type_one_list = []
    time_in_system_type_two_list = []
    num_of_bored_list = []
    num_of_bored_type_one_list = []
    num_of_bored_type_two_list = []
    time_in_q_list = []
    time_in_q_type_one_list = []
    time_in_q_type_two_list = []

    for p in patients:
        if p.bored:
            num_of_bored += 1
            time_in_system += p.deadline - p.scheduler_arrival

            time_in_system_list.append(p.deadline - p.scheduler_arrival)
            time_in_q_list.append(p.start - p.scheduler_arrival)

            time_in_q += p.deadline
        else:
            time_in_system += p.departure - p.scheduler_arrival
            time_in_q += p.start - p.scheduler_arrival

            time_in_system_list.append(p.deadline)
            time_in_q_list.append(p.deadline)

    for p in patients_type_one:
        if p.bored:
            num_of_bored_type_one += 1
            time_in_system_type_one += p.deadline
            time_in_q_type_one += p.deadline

            time_in_system_type_one_list.append(p.deadline)
            time_in_q_type_one_list.append(p.deadline)
        else:
            time_in_system_type_one += p.departure - p.scheduler_arrival
            time_in_q_type_one += p.start - p.scheduler_arrival
            time_in_system_type_one_list.append(p.deadline)
            time_in_q_type_one_list.append(p.deadline)

    for p in patients_type_two:
        if p.bored:
            num_of_bored_type_two += 1
            time_in_system_type_two += p.deadline
            time_in_q_type_two += p.deadline
            time_in_system_type_two_list.append(p.deadline)
            time_in_q_type_two_list.append(p.deadline)
        else:
            time_in_system_type_two += p.departure - p.scheduler_arrival
            time_in_q_type_two += p.start - p.scheduler_arrival
            time_in_system_type_two_list.append(p.deadline)
            time_in_q_type_two_list.append(p.deadline)

    time_in_room_q = [0 for _ in range(m)]
    for patient in patients:
        if patient.room is None:
            pass
        else:
            if patient.bored:
                time_in_room_q[rooms_list.index(patient.room)] += (patient.deadline - patient.scheduler_departure)
            else:
                time_in_room_q[rooms_list.index(patient.room)] += (patient.start - patient.scheduler_departure)

    time_in_scheduler_q = 0
    for patient in patients:
        if patient.scheduler_departure is not None:
            time_in_scheduler_q += patient.scheduler_departure - patient.scheduler_arrival
        else:
            time_in_scheduler_q += patient.deadline - patient.scheduler_arrival

    time_in_system_list.append(time_in_system)

    return True

    print("Average Time Spent in System: ", time_in_system / len(patients))
    print("Average Time Spent in System for Type One: ", time_in_system_type_one / len(patients_type_one))
    print("Average Time Spent in System for Type Two: ", time_in_system_type_two / len(patients_type_two))

    print("Average Time Spent in Queue: ", time_in_q / len(patients))
    print("Average Time Spent in Queue for Type One: ", time_in_q_type_one / len(patients_type_one))
    print("Average Time Spent in Queue for Type Two: ", time_in_q_type_two / len(patients_type_two))

    print("Average Num of bored: ", num_of_bored / len(patients))
    print("Average Num of bored for Type One: ", num_of_bored_type_one / len(patients_type_one))
    print("Average Num of bored for Type Two: ", num_of_bored_type_two / len(patients_type_one))

    print("Average Length of Scheduler Queue: ", time_in_scheduler_q)
    for i in range(m):
        print("Average Length of room" + str(i + 1) + " :" + str(time_in_room_q[i]))


if __name__ == '__main__':
    sim = Simulation(10 ** 7, Simulation.get_input_dict())
    sim.simulate()
