import numpy as np

K_E: float = 8.9875517923e9

_CHUNK = 2000


def compute_potential(
    points: np.ndarray,
    charge_centers: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    scalar = points.ndim == 1
    pts = np.atleast_2d(points)
    M = len(pts)
    phi_out = np.zeros(M)

    for start in range(0, M, _CHUNK):
        end = min(start + _CHUNK, M)
        diff = pts[start:end, np.newaxis, :] - charge_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        dist = np.maximum(dist, 1e-12)
        phi_out[start:end] = K_E * (charges[np.newaxis, :] / dist).sum(axis=-1)

    return phi_out[0] if scalar else phi_out


def compute_field(
    points: np.ndarray,
    charge_centers: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    scalar = points.ndim == 1
    pts = np.atleast_2d(points)
    M = len(pts)
    E_out = np.zeros((M, 3))

    for start in range(0, M, _CHUNK):
        end = min(start + _CHUNK, M)
        diff = pts[start:end, np.newaxis, :] - charge_centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        dist = np.maximum(dist, 1e-12)
        w = charges[np.newaxis, :] / dist ** 3
        E_out[start:end] = K_E * (w[:, :, np.newaxis] * diff).sum(axis=1)

    return E_out[0] if scalar else E_out


def compute_on_grid_2d(
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    y_slice: float,
    charge_centers: np.ndarray,
    charges: np.ndarray,
):
    X, Z = np.meshgrid(x_arr, z_arr)
    shape = X.shape
    Y = np.full(shape, y_slice)

    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    phi_flat = compute_potential(pts, charge_centers, charges)
    E_flat = compute_field(pts, charge_centers, charges)

    phi = phi_flat.reshape(shape)
    Ex = E_flat[:, 0].reshape(shape)
    Ez = E_flat[:, 2].reshape(shape)

    return phi, Ex, Ez
