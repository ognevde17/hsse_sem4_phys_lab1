import numpy as np


def make_parallel_plates(L: float, d: float, nx: int, ny: int):
    dx = L / nx
    dy = L / ny
    xs = np.linspace(-L / 2 + dx / 2, L / 2 - dx / 2, nx)
    ys = np.linspace(-L / 2 + dy / 2, L / 2 - dy / 2, ny)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel()
    yy = yy.ravel()

    n = nx * ny
    panel_area = dx * dy

    centers1 = np.column_stack([xx, yy, np.zeros(n)])
    centers2 = np.column_stack([xx, yy, np.full(n, d)])
    centers = np.vstack([centers1, centers2])
    areas = np.full(2 * n, panel_area)
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    return centers, areas, labels


def make_nested_spheres(R_inner: float, R_outer: float, n_theta: int, n_phi: int):
    cos_edges = np.linspace(1.0, -1.0, n_theta + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    d_cos = 2.0 / n_theta
    d_phi = 2.0 * np.pi / n_phi

    all_centers = []
    all_areas = []
    all_labels = []

    for label, R in enumerate([R_inner, R_outer]):
        panel_area = R ** 2 * d_cos * d_phi
        for i in range(n_theta):
            cos_mid = (cos_edges[i] + cos_edges[i + 1]) / 2.0
            sin_mid = np.sqrt(max(1.0 - cos_mid ** 2, 0.0))
            for j in range(n_phi):
                p_mid = (phi_edges[j] + phi_edges[j + 1]) / 2.0
                x = R * sin_mid * np.cos(p_mid)
                y = R * sin_mid * np.sin(p_mid)
                z = R * cos_mid
                all_centers.append([x, y, z])
                all_areas.append(panel_area)
                all_labels.append(label)

    centers = np.array(all_centers)
    areas = np.array(all_areas)
    labels = np.array(all_labels, dtype=int)

    return centers, areas, labels


def make_single_sphere(R: float, n_theta: int, n_phi: int):
    centers, areas, labels = make_nested_spheres(R, R * 10, n_theta, n_phi)
    mask = labels == 0
    return centers[mask], areas[mask], labels[mask]
