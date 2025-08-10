import torch as tc
import pickle
import matplotlib.pyplot as plt
from thermo_elastic.options import opts
from matplotlib.lines import Line2D

plt.rcParams['font.size']       = 18
plt.rcParams['lines.linewidth'] = 3.0

with open(f'./thermal/fem_simulation.p', 'rb') as f:
    
    fem_thermal_sol = pickle.load(f)
    
with open(f'./thermo_elastic/fem_thermo_elastic_simulation.p', 'rb') as f:
    
    fem_thermo_elastic_sol = pickle.load(f)
    
with open(f'./thermo_elastic/fem_mesh_motion.p', 'rb') as f:
    
    fem_mesh_motion = pickle.load(f)
    
with open(f'./thermal/fem_mesh.p', 'rb') as f:
    
    fem_thermal_mesh = pickle.load(f)
    
time    = tc.linspace(0.0, 10.0, 100)

fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

ax.axvline(x=0.0, color='k')
ax.axvline(x=opts['initial_length'], color='k')

ax.plot(fem_thermal_mesh, fem_thermal_sol[0,:], color='r', marker='o', markersize=10.0, label='TS: $t=0$ sec.')
ax.plot(fem_mesh_motion[0,:], fem_thermo_elastic_sol[0,:], color='r', marker='s', markersize=10.0, label='TES: $t=0$ sec.')

ax.plot(fem_thermal_mesh, fem_thermal_sol[50,:], color='g', marker='o', markersize=10.0, label='TS: $t=5$ sec.')
ax.plot(fem_mesh_motion[50,:], fem_thermo_elastic_sol[50,:], color='g', marker='s', markersize=10.0, label='TES: $t=5$ sec.')

ax.plot(fem_thermal_mesh, fem_thermal_sol[-1,:], color='orange', marker='o', markersize=10.0, label='TS: $t=10$ sec.')
ax.plot(fem_mesh_motion[-1,:], fem_thermo_elastic_sol[-1,:], color='orange', marker='s', markersize=10.0, label='TES: $t=10$ sec.')

fig.set_size_inches(6,6)
ax.grid(which='major', linestyle='--')
ax.minorticks_on()
ax.grid(which='minor', linestyle=':')
ax.legend()

plt.show()

print()