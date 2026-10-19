import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

theta_0 = 0.0 * np.pi
omega_0 = 0.0

g = 9.81
nu = 0.01
a = 0.8
omega = 3.0
T = 30.0
l = 1.0
h = 0.100
N = int(T / h)
tol = 1e-10
t = np.linspace(0, T, N + 1)


def pivot(t):
    return a * np.cos(omega * t)


def forcing(t):
    return -a * omega**2 * np.cos(omega * t)


def residual(theta, *, t=t, theta_0=theta_0, omega_0=omega_0):
    assert len(t) == len(theta)

    N = len(theta) - 1
    r = np.zeros_like(theta)

    a = (theta[0 : N - 1] - 2 * theta[1:N] + theta[2 : N + 1]) / h**2
    v = (theta[2 : N + 1] - theta[0 : N - 1]) / (2 * h)
    f = forcing(t)
    a_0 = g * np.sin(theta_0) + nu * omega_0 + f[0] * np.cos(theta_0)

    r[0] = theta[0] - theta_0
    r[1] = l * (theta[1] - theta[0]) / h - l * omega_0 + 0.5 * h * a_0
    r[2 : N + 1] = l * a + nu * v + g * np.sin(theta[1:N]) + f[1:N] * np.cos(theta[1:N])

    return r


def jacobian(theta, *, t=t):
    # dr / dtheta

    assert len(t) == len(theta)

    N = len(theta) - 1
    f = forcing(t)

    # J[0, 0] = 1.0
    # J[1, 0] = -l / h
    # J[1, 1] = l / h
    # J[2, 0] = 1.0 / h**2
    # J[2, 1] = -2.0 / h**2 + g / l * np.cos(theta[1])
    # J[2, 2] = 1.0 / h**2
    # etc.

    r = np.arange(2, N + 1)
    row_idx = np.concatenate([[0, 1, 1], np.repeat(r, 3)])
    col_idx = np.concatenate([[0, 0, 1], np.stack([r - 2, r - 1, r]).T.flatten()])
    values = np.zeros(3 * N)
    values[:3] = [1.0, -l / h, l / h]

    for i in range(2, N + 1):

        values[3 * i - 3] = l / h**2 - nu / (2 * h)
        values[3 * i - 2] = (
            -2.0 * l / h**2 + g * np.cos(theta[i - 1]) - f[i - 1] * np.sin(theta[i - 1])
        )
        values[3 * i - 1] = l / h**2 + nu / (2 * h)

    return csc_array((values, (row_idx, col_idx)), shape=(N + 1, N + 1))


def newton_raphson(*, theta_init=None, t=t, theta_0=theta_0, omega_0=omega_0):
    if theta_init is None:
        theta = np.zeros_like(t)
    else:
        theta = theta_init

    assert len(t) == len(theta)

    for it in range(100):
        r = residual(theta, t=t, theta_0=theta_0, omega_0=omega_0)

        print(np.linalg.norm(r))

        if np.linalg.norm(r) < tol * N:
            break

        # theta += np.linalg.solve(jacobian(theta, t=t), -r)
        theta += spsolve(jacobian(theta, t=t), -r)
    else:
        raise RuntimeError("NOT CONVERGED!")

    return theta


N_chunk = int(1 / h)

ts = []
thetas = []
theta_full = np.zeros(N + 1)

for i in range(int(T)):
    if len(thetas) == 0:
        theta_init = theta_0
        omega_init = omega_0
    else:
        theta_init = thetas[-1][-1]
        omega_init = (thetas[-1][-1] - thetas[-1][-2]) / h

    idx = (i * N_chunk, (i + 1) * N_chunk + 1)
    t_chunk = t[idx[0] : idx[1]]
    theta_chunk = newton_raphson(t=t_chunk, theta_0=theta_init, omega_0=omega_init)

    ts.append(t_chunk)
    thetas.append(theta_chunk)
    theta_full[idx[0] : idx[1]] = theta_chunk

theta = newton_raphson(theta_init=theta_full)

plt.figure()
for t_chunk, theta_chunk in zip(ts, thetas):
    plt.plot(t_chunk, theta_chunk)
plt.show()

plt.figure()
plt.plot(t, theta)
plt.show()


fps = min(50, int(1 / h))
frames = int(T * fps)

from matplotlib.patches import Circle

fig, ax = plt.subplots(layout="tight")

x_pivot = pivot(t[0])
y_pivot = 0.0
x_mass = x_pivot + l * np.sin(theta[0])
y_mass = y_pivot - l * np.cos(theta[0])

scat = ax.scatter(x_pivot, y_pivot, c="k", marker="o")
line = ax.plot([x_pivot, x_mass], [y_pivot, y_mass], c="k")[0]
patch = ax.add_patch(Circle([x_mass, y_mass], 0.15, ec="k", fc="0.5", zorder=2))

ax.set_xlim([-2.0, 2.0])
ax.set_ylim([-2.0, 2.0])
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_aspect("equal")


def update(frame):
    idx = np.sum(t < frame / fps)
    x_pivot = pivot(t[idx])
    y_pivot = 0.0
    x_mass = x_pivot + l * np.sin(theta[idx])
    y_mass = y_pivot - l * np.cos(theta[idx])

    scat.set_offsets([pivot(t[idx]), 0.0])
    line.set_xdata([x_pivot, x_mass])
    line.set_ydata([y_pivot, y_mass])
    patch.center = [x_mass, y_mass]

    return scat, line, patch


ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=1000.0 / fps)
fname = "pendulum_a-{:.3f}_omega-{:.3f}_h-{:.3f}.mp4".format(a, omega, h)
ani.save(filename=fname, writer="ffmpeg")

plt.show()
