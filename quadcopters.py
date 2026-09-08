import numpy as np
from numpy.typing import NDArray

from quat_utils import *

class Quadcopter:

    def __init__(self):
        self.g = 9.81  # [m/s^2] gravity
        self.m = 0.8  # [kg] mass
        self.L = 0.5  # [m] arm length
        self.k = 100  # [] thrust / drag ratio.
        self.I = np.diag([0.001, 0.001, 0.002])  # noqa: E741  # inertia matrix.
        self.I_inv = np.linalg.inv(self.I)

        self.state = np.zeros(13)

    def mixer(self, u: NDArray) -> NDArray:
        u_c, u_p, u_q, u_r = u
        F_t = u_c * self.m / 4
        F1 = F_t - u_q / (2 * self.L) + u_r / (4 * self.k)
        F2 = F_t + u_p / (2 * self.L) - u_r / (4 * self.k)
        F3 = F_t + u_q / (2 * self.L) + u_r / (4 * self.k)
        F4 = F_t - u_p / (2 * self.L) - u_r / (4 * self.k)
        return np.array([F1, F2, F3, F4])

    def dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        F1, F2, F3, F4 = self.mixer(u)

        q, v, omega = x[3:7], x[7:10], x[10:13]
        c = np.array([0, 0, (F1 + F2 + F3 + F4) / self.m])
        tau1, tau2, tau3, tau4 = F1 / self.k, F2 / self.k, F3 / self.k, F4 / self.k
        Tau = np.array([self.L * (F2 - F4), self.L * (F3 - F1), tau1 - tau2 + tau3 - tau4])

        return np.concat(
            [
                v,
                0.5 * quat_mul(q, np.array([0, *omega])),
                np.array([0, 0, -self.g]) + rot_vec_by_quat(q, c),
                self.I_inv @ (Tau - np.cross(omega, self.I @ omega)),
            ]
        )

    def step(self, dt, u):
        self.state += self.dynamics(self.state, u) * dt

        # Quaternion drifts under Euler integration, so we renormalize it.
        q = self.state[3:7]
        self.state[3:7] = q / np.linalg.norm(q)
