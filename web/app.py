import sys
import os
import io
import base64
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import make_parallel_plates, make_nested_spheres
from mom_solver import solve_charges, compute_capacitance
from visualization import plot_plates, plot_spheres

app = Flask(__name__)

EPS_0 = 8.854187817e-12


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True)
    geometry = data.get("geometry", "plates")

    try:
        if geometry == "plates":
            return _run_plates(data)
        elif geometry == "spheres":
            return _run_spheres(data)
        else:
            return jsonify({"error": "Неизвестная геометрия"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc), "detail": traceback.format_exc()}), 500


def _run_plates(data: dict):
    L  = float(data["L"])
    d  = float(data["d"])
    nx = int(data["nx"])
    ny = int(data["ny"])
    V  = float(data["V"])

    if L <= 0 or d <= 0:
        return jsonify({"error": "L и d должны быть положительными"}), 400
    if d >= L * 10:
        return jsonify({"error": "d слишком велико относительно L"}), 400
    if not (2 <= nx <= 30 and 2 <= ny <= 30):
        return jsonify({"error": "nx и ny: от 2 до 30"}), 400
    if V == 0:
        return jsonify({"error": "V не может быть равно 0"}), 400

    centers, areas, labels = make_parallel_plates(L, d, nx, ny)
    potentials = {0: V / 2, 1: -V / 2}
    q = solve_charges(centers, areas, labels, potentials)
    C = compute_capacitance(q, labels, V)

    C_inf = EPS_0 * L ** 2 / d
    Q_pos = float(np.sum(q[labels == 0]))
    Q_neg = float(np.sum(q[labels == 1]))

    plot_b64 = _fig_to_b64(plot_plates, centers, q, labels, L, d, C)

    return jsonify({
        "C_mom":        round(C * 1e12, 4),
        "C_analytical": round(C_inf * 1e12, 4),
        "C_ratio":      round(C / C_inf, 4),
        "Q_pos":        round(Q_pos * 1e12, 6),
        "Q_neg":        round(Q_neg * 1e12, 6),
        "Q_total":      round((Q_pos + Q_neg) * 1e15, 3),
        "N_panels":     len(centers),
        "plot":         plot_b64,
    })


def _run_spheres(data: dict):
    R_inner = float(data["R_inner"])
    R_outer = float(data["R_outer"])
    n_theta = int(data["n_theta"])
    n_phi   = int(data["n_phi"])
    V       = float(data["V"])

    if R_inner <= 0 or R_outer <= R_inner:
        return jsonify({"error": "R_outer должен быть больше R_inner > 0"}), 400
    if not (4 <= n_theta <= 30):
        return jsonify({"error": "n_theta: от 4 до 30"}), 400
    if not (8 <= n_phi <= 60):
        return jsonify({"error": "n_phi: от 8 до 60"}), 400
    if V == 0:
        return jsonify({"error": "V не может быть равно 0"}), 400

    centers, areas, labels = make_nested_spheres(R_inner, R_outer, n_theta, n_phi)
    potentials = {0: V / 2, 1: -V / 2}
    q = solve_charges(centers, areas, labels, potentials)
    C = compute_capacitance(q, labels, V)

    C_exact = 4 * np.pi * EPS_0 * R_inner * R_outer / (R_outer - R_inner)
    rel_err = abs(C - C_exact) / C_exact * 100
    Q_pos   = float(np.sum(q[labels == 0]))
    Q_neg   = float(np.sum(q[labels == 1]))

    plot_b64 = _fig_to_b64(plot_spheres, centers, q, labels, R_inner, R_outer, C)

    return jsonify({
        "C_mom":        round(C * 1e12, 4),
        "C_analytical": round(C_exact * 1e12, 4),
        "rel_err":      round(rel_err, 3),
        "Q_pos":        round(Q_pos * 1e12, 6),
        "Q_neg":        round(Q_neg * 1e12, 6),
        "Q_total":      round((Q_pos + Q_neg) * 1e15, 3),
        "N_panels":     len(centers),
        "plot":         plot_b64,
    })


def _fig_to_b64(plot_fn, *args, **kwargs) -> str:
    buf = io.BytesIO()
    plot_fn(*args, filename=buf, **kwargs)
    return base64.b64encode(buf.read()).decode("utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"  Открой браузер: http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
