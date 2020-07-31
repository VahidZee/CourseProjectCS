from main import Simulation
import numpy as np


# a test setting
hparams = {
    'rooms': [
        np.array([5, 5, 5], np.double),
        np.array([5, 5, 5], np.double),

    ],
    'max_patient_wait': np.array([5], dtype=np.double),
    'arrival_rate': np.array([0.1], dtype=np.double),
    'reception_mu': np.array([0.2], dtype=np.double)
}

sim = Simulation(10**4, hparams)

sim.simulate()
print(sim.reception)
sim.reception.plot_room_history()
print(sum(sim.reception.wait_history[0])/len(sim.reception.wait_history[0]))
q = sim.reception
t = np.array(q.len_history_timestamps).reshape(-1)
t[1:] = t[1:] - t[:len(t)-1]
print((np.array(q.len_history) * t).sum(1) / t.sum())
print(t)
