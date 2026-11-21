import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from tcc import tcc


# %% Synthetic flow
x = np.linspace(0, 8, 48)
y = np.linspace(-2, 2, 28)
xx, yy = np.meshgrid(x, y)
dt = 0.05
time = np.arange(1200) * dt
r = 10
K = 3
lags = np.arange(80)
rng = np.random.default_rng(20260829)

# A, k, Omega, phase, xc, yc, Lx, Ly
parameters = np.array([
    [3.00, 0.70, 0.80, 0.10, 3.00, 0.00, 2.60, 1.05],
    [2.20, 2.20, 3.60, 0.55, 5.00, 1.05, 1.70, 0.42],
    [1.00, 1.40, 1.40, 0.25, 4.70, 0.00, 2.20, 0.42],
    [0.80, 2.60, 2.20, -0.35, 3.80, -0.58, 1.80, 0.28],
    [0.65, 3.50, 2.90, 0.70, 5.70, 0.35, 1.30, 0.30],
])

flow = np.zeros((2 * xx.size, time.size))
target_flow = np.zeros_like(flow)
for j, (A, k, omega, phase, xc, yc, Lx, Ly) in enumerate(parameters):
    envelope = np.exp(-((xx - xc) / Lx) ** 2 - ((yy - yc) / Ly) ** 2)
    psi = A * envelope * np.exp(1j * (k * xx + phase))
    u_mode = -2 * (yy - yc) / Ly ** 2 * psi
    v_mode = (2 * (xx - xc) / Lx ** 2 - 1j * k) * psi
    mode = np.concatenate((u_mode.ravel(), v_mode.ravel()))
    component = np.real(mode[:, None] * np.exp(-1j * omega * time))
    flow += component
    if j >= 2:
        target_flow += component

noise_std = 0.08 * np.sqrt(np.mean(flow ** 2))
state = flow + noise_std * rng.standard_normal(flow.shape)
state -= state.mean(axis=1, keepdims=True)
observable = (
    np.cos(1.40 * (time - 0.35))
    + 0.8 * np.sin(2.20 * (time - 0.55))
    + 0.6 * np.cos(2.90 * (time - 0.25))
    + 0.04 * rng.standard_normal(time.size)
)


# %% DMD and TCC
X1 = state[:, :-1]
X2 = state[:, 1:]
U, singular_values, Vh = svds(
    X1, k=r, solver='propack', rng=np.random.default_rng(20260907)
)
svd_order = np.argsort(singular_values)[::-1]
U = U[:, svd_order]
singular_values = singular_values[svd_order]
V = Vh[svd_order].conj().T
projected_X2 = (X2 @ V) / singular_values
eigenvalues, eigenvectors = np.linalg.eig(U.conj().T @ projected_X2)
phi = projected_X2 @ eigenvectors
phi /= np.linalg.norm(phi, axis=0)
coefficient = np.linalg.lstsq(phi, state, rcond=None)[0]

angular_frequency = np.imag(np.log(eigenvalues.astype(complex))) / dt
positive = np.flatnonzero(angular_frequency > 0)
positive = positive[np.argsort(angular_frequency[positive])]
groups = np.array([
    [j, np.argmin(np.abs(eigenvalues - eigenvalues[j].conjugate()))]
    for j in positive
])
frequency = angular_frequency[positive]
_, order = tcc(phi, coefficient, observable, groups, lags)

selected = groups[order[:K]].ravel()
reconstruction = np.real(phi[:, selected] @ coefficient[selected])
field_error = 100 * np.linalg.norm(target_flow - reconstruction) / np.linalg.norm(target_flow)

print('TCC Top-3 angular frequencies:', np.round(frequency[order[:K]], 4))
print(f'Target-subfield reconstruction error: {field_error:.2f}%')


# %% Plotting
plt.rcParams['font.family'] = 'Times New Roman'
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

snapshot = 160
truth_v = target_flow[xx.size:, snapshot].reshape(xx.shape)
reconstructed_v = reconstruction[xx.size:, snapshot].reshape(xx.shape)
limit = np.max(np.abs(truth_v))
for ax, field, title in zip(
    axes, [truth_v, reconstructed_v], ['True target subfield', 'TCC Top-3 subfield']
):
    image = ax.pcolormesh(xx, yy, field, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
    ax.set(xlabel='x', ylabel='y', title=f'{title} at t = {time[snapshot]:.2f}')
    fig.colorbar(image, ax=ax, label='v')

plt.tight_layout()
plt.savefig('synthetic_result.png', dpi=300)
plt.show()
