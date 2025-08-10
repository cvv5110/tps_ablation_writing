import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# --- Physical and Mesh Parameters ---
L0 = 0.02
N = 50
x_ref = np.linspace(0, L0, N + 1)

rho = 4500
cp = 522
k = 21.9

# --- Time Parameters ---
dt = 0.01
t_final = 10.0
n_steps = int(t_final / dt)
times = np.linspace(0, t_final, n_steps + 1)

# --- Initial and Boundary Conditions ---
T_init = 300.0
q_left = 1e5

def v_m(t):
    return 1e-4

def u_field(x, t):
    return v_m(t) * t * (1 - x / L0)

def v_field(x, t):
    return v_m(t) * (1 - x / L0)

# --- Quadrature and Basis Functions ---
quad_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
quad_wts = np.array([1.0, 1.0])

def phi(xi):
    return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])

def assemble_mass_stiffness_dynamic(x_mesh):
    M = lil_matrix((N + 1, N + 1))
    K = lil_matrix((N + 1, N + 1))
    for e in range(N):
        i, j = e, e + 1
        dx_e = x_mesh[j] - x_mesh[i]
        Me = (rho * cp * dx_e / 6) * np.array([[2, 1], [1, 2]])
        Ke = (k / dx_e) * np.array([[1, -1], [-1, 1]])
        M[i:i+2, i:i+2] += Me
        K[i:i+2, i:i+2] += Ke
    return M.tocsc(), K.tocsc()

def compute_advection_matrix(t, x_mesh):
    C = lil_matrix((N + 1, N + 1))
    for e in range(N):
        i, j = e, e + 1
        x_e = x_mesh[i]
        x_ep1 = x_mesh[j]
        dx_e = x_ep1 - x_e
        J_e = dx_e / 2
        dphi_dx = np.array([-1 / dx_e, 1 / dx_e])
        C_local = np.zeros((2, 2))
        for q in range(len(quad_pts)):
            xi_q = quad_pts[q]
            w_q = quad_wts[q]
            x_q = (x_e + x_ep1) / 2 + J_e * xi_q
            phi_q = phi(xi_q)
            v_xq = v_field(x_q, t)
            for m in range(2):
                for n in range(2):
                    C_local[m, n] += w_q * v_xq * dphi_dx[n] * phi_q[m] * J_e
        C_local *= rho * cp
        C[i:i+2, i:i+2] += C_local
    return C.tocsc()

# --- Initialize Temperature ---
T = np.ones(N + 1) * T_init
T_hist = [T.copy()]
x_hist = [x_ref.copy()]

# --- Time-Stepping Loop ---
for step in range(n_steps):
    t = times[step]
    x_mesh = x_ref + u_field(x_ref, t)
    M, K = assemble_mass_stiffness_dynamic(x_mesh)
    C = compute_advection_matrix(t, x_mesh)
    A = M + dt * (K - C)
    b = M @ T
    b[0] += dt * q_left
    T = spsolve(A, b)
    T_hist.append(T.copy())
    x_hist.append(x_mesh.copy())

# --- Plot Results ---
T_hist = np.array(T_hist)
x_hist = np.array(x_hist)
snapshots = [0, n_steps // 2, n_steps]
labels = ['t = 0 s', f't = {t_final/2:.1f} s', f't = {t_final:.1f} s']
colors = ['b', 'g', 'r']

plt.figure(figsize=(8, 5))
for idx, label, color in zip(snapshots, labels, colors):
    plt.plot(x_hist[idx], T_hist[idx], label=label, color=color)

plt.xlabel("Deformed Mesh Position x [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature on Shrinking Mesh at Selected Time Snapshots")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print()