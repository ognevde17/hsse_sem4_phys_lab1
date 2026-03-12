from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

from field import compute_on_grid_2d


def plot_plates(
    charge_centers: np.ndarray,
    charges: np.ndarray,
    labels: np.ndarray,
    L: float,
    d: float,
    capacitance: float,
    nx_grid: int = 60,
    nz_grid: int = 60,
    filename=None,
) -> None:
    margin_x = L * 0.4
    margin_z = d * 1.0

    x_arr = np.linspace(-L / 2 - margin_x, L / 2 + margin_x, nx_grid)
    z_arr = np.linspace(-margin_z, d + margin_z, nz_grid)

    phi, Ex, Ez = compute_on_grid_2d(x_arr, z_arr, 0.0, charge_centers, charges)

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    vmax = np.abs(phi).max()
    cf = ax1.contourf(x_arr, z_arr, phi, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.contour(x_arr, z_arr, phi, levels=20, colors="k", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax1, label="φ, В", shrink=0.85)
    _draw_plates(ax1, L, d)
    ax1.set_xlabel("x, м")
    ax1.set_ylabel("z, м")
    ax1.set_title("Потенциал")
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(gs[1])
    E_mag = np.hypot(Ex, Ez)
    E_mag_safe = np.where(E_mag > 0, E_mag, E_mag.max() * 1e-6)
    sp = ax2.streamplot(
        x_arr, z_arr, Ex, Ez,
        color=np.log1p(E_mag_safe),
        cmap="plasma",
        density=1.6,
        linewidth=0.9,
        arrowsize=0.8,
    )
    fig.colorbar(sp.lines, ax=ax2, label="ln(1+|E|)", shrink=0.85)
    _draw_plates(ax2, L, d)
    ax2.set_xlabel("x, м")
    ax2.set_ylabel("z, м")
    ax2.set_title("Силовые линии поля")
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(gs[2])
    mask0 = labels == 0
    mask1 = labels == 1

    y0 = charge_centers[mask0, 1]
    y_mid = np.median(np.abs(y0))
    near_mid = np.abs(y0) <= y_mid * 1.01 + 1e-12
    idx_mid = np.argsort(charge_centers[mask0, 0][near_mid])
    x_mid = charge_centers[mask0, 0][near_mid][idx_mid]
    panel_side = L / int(np.round(np.sqrt(mask0.sum())))
    sigma0 = charges[mask0][near_mid][idx_mid] / panel_side ** 2

    y1 = charge_centers[mask1, 1]
    near_mid1 = np.abs(y1) <= np.median(np.abs(y1)) * 1.01 + 1e-12
    idx_mid1 = np.argsort(charge_centers[mask1, 0][near_mid1])
    sigma1 = charges[mask1][near_mid1][idx_mid1] / panel_side ** 2

    ax3.plot(x_mid, sigma0 * 1e9, label="Пластина 1 (+)", color="tab:red")
    ax3.plot(x_mid, sigma1 * 1e9, label="Пластина 2 (−)", color="tab:blue")
    ax3.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax3.set_xlabel("x, м")
    ax3.set_ylabel("σ, нКл/м²")
    ax3.set_title("Поверхностный заряд (y≈0)")
    ax3.legend(fontsize=8)

    fig.suptitle(
        f"Метод моментов — две параллельные пластины   "
        f"L={L} м, d={d} м,  C = {capacitance * 1e12:.3f} пФ",
        fontsize=11,
    )

    _save_or_show(fig, filename)


def _draw_plates(ax, L, d):
    for z_pos, color in [(0, "#d62728"), (d, "#1f77b4")]:
        ax.plot([-L / 2, L / 2], [z_pos, z_pos], color=color, lw=3, solid_capstyle="butt")


def plot_spheres(
    charge_centers: np.ndarray,
    charges: np.ndarray,
    labels: np.ndarray,
    R_inner: float,
    R_outer: float,
    capacitance: float,
    n_grid: int = 70,
    filename=None,
) -> None:
    margin = R_outer * 0.5
    lim = R_outer + margin

    x_arr = np.linspace(-lim, lim, n_grid)
    z_arr = np.linspace(-lim, lim, n_grid)

    phi, Ex, Ez = compute_on_grid_2d(x_arr, z_arr, 0.0, charge_centers, charges)

    X, Z = np.meshgrid(x_arr, z_arr)
    inside_inner = X ** 2 + Z ** 2 < R_inner ** 2
    outside_outer = X ** 2 + Z ** 2 > R_outer ** 2

    phi_masked = np.where(inside_inner | outside_outer, np.nan, phi)
    E_mag = np.hypot(Ex, Ez)
    E_masked = np.where(inside_inner | outside_outer, np.nan, E_mag)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    vmax = np.nanmax(np.abs(phi_masked))
    cf = ax.contourf(x_arr, z_arr, phi_masked, levels=20, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    ax.contour(x_arr, z_arr, phi_masked, levels=20, colors="k",
               linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, label="φ, В", shrink=0.85)
    _draw_circles(ax, R_inner, R_outer)
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.set_title("Потенциал")
    ax.set_aspect("equal")

    ax = axes[1]
    Ex_plot = np.where(inside_inner | outside_outer, 0.0, Ex)
    Ez_plot = np.where(inside_inner | outside_outer, 0.0, Ez)
    E_safe = np.where(E_masked > 0, E_masked, np.nanmax(E_masked) * 1e-6)
    sp = ax.streamplot(
        x_arr, z_arr, Ex_plot, Ez_plot,
        color=np.log1p(np.where(np.isnan(E_safe), 0, E_safe)),
        cmap="plasma",
        density=1.4,
        linewidth=0.9,
        arrowsize=0.8,
    )
    fig.colorbar(sp.lines, ax=ax, label="ln(1+|E|)", shrink=0.85)
    _draw_circles(ax, R_inner, R_outer)
    ax.set_xlabel("x, м")
    ax.set_ylabel("z, м")
    ax.set_title("Силовые линии поля")
    ax.set_aspect("equal")

    fig.suptitle(
        f"Метод моментов — концентрические сферы   "
        f"R₁={R_inner} м, R₂={R_outer} м,  C = {capacitance * 1e12:.3f} пФ",
        fontsize=11,
    )

    _save_or_show(fig, filename)


def _draw_circles(ax, R_inner, R_outer):
    for R, color in [(R_inner, "#d62728"), (R_outer, "#1f77b4")]:
        circle = Circle((0, 0), R, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(circle)


def _save_or_show(fig: plt.Figure, filename) -> None:
    if filename is None:
        plt.show()
    elif hasattr(filename, "write"):
        fig.savefig(filename, dpi=150, bbox_inches="tight", format="png")
        if hasattr(filename, "seek"):
            filename.seek(0)
    else:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  График сохранён: {filename}")
    plt.close(fig)
