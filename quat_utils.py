import numpy as np
from numpy.typing import NDArray

def quaternion_to_euler(q: NDArray) -> tuple[float, float, float]:
    w, qx, qy, qz = q
    phi = np.arctan2(2 * (w * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    theta = np.arcsin(2 * (w * qy - qz * qx))
    psi = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return float(phi), float(theta), float(psi)

def rot_vec_by_quat(q: NDArray, v: NDArray) -> NDArray:
    w = q[0]
    q_vec = q[1:4]
    t = 2.0 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)

def quat_mul(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    pw, px, py, pz = p
    qw, qx, qy, qz = q

    r = np.empty(4, dtype=np.float64)

    r[0] = pw * qw - px * qx - py * qy - pz * qz
    r[1] = pw * qx + px * qw + py * qz - pz * qy
    r[2] = pw * qy - px * qz + py * qw + pz * qx
    r[3] = pw * qz + px * qy - py * qx + pz * qw

    return r
