import numpy as np
import matplotlib.pyplot as plt

theta_0 = 0.0 * np.pi
omega_0 = 0.0

g = 9.81
nu = 0.01
T = 30.0
l = 1.0
N = 1000
h = T / N
tol = 1e-8


def forcing(t):
    # return 0.0

    a = 0.5
    omega = 3.0

    return -a * omega**2 * np.cos(omega * t)


def residual(theta, t=np.linspace(0, T, N + 1)):
    r = np.zeros(N + 1)

    a = (theta[0 : N - 1] - 2 * theta[1:N] + theta[2 : N + 1]) / h**2
    omega = (theta[2 : N + 1] - theta[0 : N - 1]) / (2 * h)
    f = forcing(t)
    a_0 = g * np.sin(theta_0) + nu * omega_0 + f[0] * np.cos(theta_0)

    r[0] = theta[0] - theta_0
    r[1] = l * (theta[1] - theta[0]) / h - l * omega_0 + 0.5 * h * a_0
    # r[2] = (theta[0] - 2 * theta[1] + theta[2]) / h**2 + g / l * np.sin(theta[1])
    r[2 : N + 1] = l * a + nu * omega + g * np.sin(theta[1:N]) + f[1:N] * np.cos(theta[1:N])

    return r


def jacobian(theta, t=np.linspace(0, T, N + 1)):
    # dr / dtheta

    f = forcing(t)

    J = np.zeros((N + 1, N + 1))

    J[0, 0] = 1.0
    J[1, 0] = -l / h
    J[1, 1] = l / h

    # J[2, 0] = 1.0 / h**2
    # J[2, 1] = -2.0 / h**2 + g / l * np.cos(theta[1])
    # J[2, 2] = 1.0 / h**2

    for i in range(2, N + 1):
        J[i, i - 2] = l / h**2 - nu / (2 * h)
        J[i, i - 1] = -2.0 * l / h**2 + g * np.cos(theta[i - 1]) - f[i - 1] * np.sin(theta[i - 1])
        J[i, i] = l / h**2 + nu / (2 * h)

    return J


theta = theta_0 * np.ones(N + 1)

for it in range(100):
    r = residual(theta)

    print(np.linalg.norm(r))

    if np.linalg.norm(r) < tol * N:
        break

    theta += np.linalg.solve(jacobian(theta), -r)
else:
    print("NOT CONVERGED!")

plt.figure()
plt.plot(np.linspace(0, T, N + 1), theta)
plt.show()
