import numpy as np
from scipy import linalg

K_E: float = 8.9875517923e9


def build_potential_matrix(centers: np.ndarray, areas: np.ndarray) -> np.ndarray:
    N = len(centers)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    np.fill_diagonal(dist, 1.0)
    P = K_E / dist

    self_coeffs = K_E * 2.0 * np.sqrt(np.pi / areas)
    np.fill_diagonal(P, self_coeffs)

    return P


def solve_charges(
    centers: np.ndarray,
    areas: np.ndarray,
    labels: np.ndarray,
    potentials: dict,
) -> np.ndarray:
    phi_target = np.array([potentials[int(lbl)] for lbl in labels])
    P = build_potential_matrix(centers, areas)
    N = len(centers)

    if len(np.unique(labels)) > 1:
        q0 = linalg.solve(P, phi_target)
        p = linalg.solve(P, np.ones(N))
        lam = np.sum(q0) / np.sum(p)
        q = q0 - lam * p
    else:
        q = linalg.solve(P, phi_target)

    return q


def compute_capacitance(
    charges: np.ndarray,
    labels: np.ndarray,
    voltage: float,
    positive_label: int = 0,
) -> float:
    Q = float(np.sum(charges[labels == positive_label]))
    return Q / voltage
