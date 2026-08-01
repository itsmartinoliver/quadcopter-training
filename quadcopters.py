import numpy as np
from numpy.typing import NDArray

# https://mrandri19.github.io/2026/04/03/2d-quadcopter-simulation.html
class Quadcopter:
    def __init__(self, m, l, I):
        self.g, self.m, self.l, self.I = 9.81, m, l, I
        self.state = np.zeros(shape=(6,), dtype=np.float32) # State

    def step(self, dt, F: NDArray) -> NDArray:
        u = [F[0] + F[1], F[0] - F[1]]
        _y, _z, phi, y_dot, z_dot, phi_dot = self.state
        self.state += dt * np.array( # Update state
            [
                y_dot,
                z_dot,
                phi_dot,
                -u[0] / self.m * np.sin(phi),
                u[0] / self.m * np.cos(phi) - self.g,
                u[1] / self.I * self.l,
            ], dtype=np.float32
        )
