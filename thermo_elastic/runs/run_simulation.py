from thermo_elastic import fem_thermo_elastic_solver
from thermo_elastic.options import opts
import torch as tc
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import pickle

solver = fem_thermo_elastic_solver(options=opts)
time   = tc.linspace(0.0, 10.0, 100)
T_0    = solver.initial_temperature * tc.ones((solver.number_nodes,))

sol = odeint(solver.rpm, T_0, time)

# compute mesh motion

mesh = []

for _t in time:
    
    _u      = solver._displacement_field(_t, solver.initial_mesh)
    _x_mesh = solver.initial_mesh + _u
    mesh.append(_x_mesh)

mesh = tc.stack(mesh)

with open(f'./thermo_elastic/fem_thermo_elastic_simulation.p', 'wb') as f:
    
    pickle.dump(sol, f)
    
with open(f'./thermo_elastic/fem_mesh_motion.p', 'wb') as f:
    
    pickle.dump(mesh, f)

print()