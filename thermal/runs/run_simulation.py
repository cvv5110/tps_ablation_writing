from thermal import fem_thermal_solver
from thermal import fdm_thermal_solver
from thermal.options import opts
from torchdiffeq import odeint
import torch as tc
import matplotlib.pyplot as plt
import pickle

fem_solver = fem_thermal_solver(options=opts)
fdm_solver = fdm_thermal_solver(options=opts)
time       = tc.linspace(0.0, 10.0, 100)
To         = fem_solver.initial_temperature * tc.ones((fem_solver.number_elements+1,))

# fem simulation
fem_sol    = odeint(fem_solver.rpm, To, time)

# fdm simulation
fdm_sol = []
T       = To
nt      = 10_000

for _n in range(nt):
    
    if _n % (nt // 10) == 0 or _n == (nt - 1):
        
        fdm_sol.append(T)
    
    T = fdm_solver.A @ T + fdm_solver.b
    
fdm_sol = tc.stack(fdm_sol)

fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
fig.set_size_inches(6,6)
ax.plot(fem_solver.x, fem_sol.T, color='k', label='FEM')
ax.plot(fdm_solver.x, fdm_sol.T, color='orange', linestyle='dashdot', label='FDM')
ax.grid()
ax.set_xlabel('Space, [m]')
ax.set_ylabel('Temperature, [K]')
plt.show()

fig.savefig(f'./thermal/figures/fem_fdm_simulation.png', dpi=600)

with open(f'./thermal/fem_simulation.p', 'wb') as f:
    
    pickle.dump(fem_sol, f)
    
with open(f'./thermal/fdm_simulation.p', 'wb') as f:
    
    pickle.dump(fdm_sol, f)
    
with open(f'./thermal/fem_mesh.p', 'wb') as f:
    
    pickle.dump(fem_solver.x, f)
    
print()
