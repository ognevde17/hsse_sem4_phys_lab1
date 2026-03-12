import argparse
import numpy as np

from geometry import make_parallel_plates, make_nested_spheres
from mom_solver import solve_charges, compute_capacitance
from visualization import plot_plates, plot_spheres

EPS_0 = 8.854187817e-12


def run_plates(args) -> None:
    print(f"\n--- Параллельные пластины ---")
    print(f"  L = {args.L} м,  d = {args.d} м,  {args.nx}×{args.ny} панелей на пластину")
    print(f"  V = {args.V} В")

    centers, areas, labels = make_parallel_plates(args.L, args.d, args.nx, args.ny)
    potentials = {0: args.V / 2, 1: -args.V / 2}
    print(f"  Всего панелей: {len(centers)}")

    print("  Решение системы уравнений ... ", end="", flush=True)
    q = solve_charges(centers, areas, labels, potentials)
    print("готово.")

    C = compute_capacitance(q, labels, args.V)
    C_inf = EPS_0 * args.L ** 2 / args.d
    Q_pos = np.sum(q[labels == 0])
    Q_neg = np.sum(q[labels == 1])

    print(f"\n  Ёмкость (МоМ):            C = {C * 1e12:.4f} пФ")
    print(f"  Ёмкость (бесконечные):    C = {C_inf * 1e12:.4f} пФ")
    print(f"  Отношение C/C_inf:            {C / C_inf:.4f}  (>1 из-за краевых эффектов)")
    print(f"  Заряд электрода +: Q = {Q_pos * 1e12:.4f} пКл")
    print(f"  Заряд электрода −: Q = {Q_neg * 1e12:.4f} пКл")
    print(f"  Нейтральность: Q+ + Q− = {(Q_pos + Q_neg):.2e} Кл")

    out = args.output or "output_plates.png"
    plot_plates(centers, q, labels, args.L, args.d, C, filename=out)


def run_spheres(args) -> None:
    print(f"\n--- Концентрические сферы ---")
    print(f"  R_inner = {args.R_inner} м,  R_outer = {args.R_outer} м")
    print(f"  Разбиение: {args.n_theta}×{args.n_phi}")
    print(f"  V = {args.V} В")

    centers, areas, labels = make_nested_spheres(
        args.R_inner, args.R_outer, args.n_theta, args.n_phi
    )
    potentials = {0: args.V / 2, 1: -args.V / 2}
    print(f"  Всего панелей: {len(centers)}")

    print("  Решение системы уравнений ... ", end="", flush=True)
    q = solve_charges(centers, areas, labels, potentials)
    print("готово.")

    C = compute_capacitance(q, labels, args.V)
    a, b = args.R_inner, args.R_outer
    C_exact = 4 * np.pi * EPS_0 * a * b / (b - a)
    Q_pos = np.sum(q[labels == 0])
    Q_neg = np.sum(q[labels == 1])

    print(f"\n  Ёмкость (МоМ):        C = {C * 1e12:.4f} пФ")
    print(f"  Ёмкость (точная):     C = {C_exact * 1e12:.4f} пФ")
    print(f"  Отн. погрешность:       {abs(C - C_exact) / C_exact * 100:.2f}%")
    print(f"  Заряд внутренней: Q = {Q_pos * 1e12:.4f} пКл")
    print(f"  Заряд внешней:    Q = {Q_neg * 1e12:.4f} пКл")
    print(f"  Нейтральность: Q+ + Q− = {(Q_pos + Q_neg):.2e} Кл")

    out = args.output or "output_spheres.png"
    plot_spheres(centers, q, labels, args.R_inner, args.R_outer, C, filename=out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Метод моментов для основной задачи электростатики (М1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="geometry", required=True, help="Тип геометрии")

    p_plates = sub.add_parser("plates", help="Две параллельные пластины")
    p_plates.add_argument("--L", type=float, default=1.0, help="Сторона квадратной пластины [м]")
    p_plates.add_argument("--d", type=float, default=0.2, help="Расстояние между пластинами [м]")
    p_plates.add_argument("--nx", type=int, default=12, help="Панелей по x на каждой пластине")
    p_plates.add_argument("--ny", type=int, default=12, help="Панелей по y на каждой пластине")
    p_plates.add_argument("--V", type=float, default=1.0, help="Разность потенциалов [В]")
    p_plates.add_argument("--output", type=str, default=None, help="Путь для сохранения PNG")

    p_spheres = sub.add_parser("spheres", help="Две концентрические сферы")
    p_spheres.add_argument("--R_inner", type=float, default=0.3, help="Радиус внутренней сферы [м]")
    p_spheres.add_argument("--R_outer", type=float, default=0.6, help="Радиус внешней сферы [м]")
    p_spheres.add_argument("--n_theta", type=int, default=20, help="Широтных полос")
    p_spheres.add_argument("--n_phi", type=int, default=40, help="Долготных полос")
    p_spheres.add_argument("--V", type=float, default=1.0, help="Разность потенциалов [В]")
    p_spheres.add_argument("--output", type=str, default=None, help="Путь для сохранения PNG")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.geometry == "plates":
        if args.d <= 0 or args.L <= 0:
            parser.error("L и d должны быть положительными.")
        if args.nx < 2 or args.ny < 2:
            parser.error("nx и ny должны быть не менее 2.")
        run_plates(args)
    else:
        if args.R_inner <= 0 or args.R_outer <= args.R_inner:
            parser.error("Необходимо R_outer > R_inner > 0.")
        if args.n_theta < 4 or args.n_phi < 8:
            parser.error("n_theta >= 4 и n_phi >= 8 для разумной дискретизации.")
        run_spheres(args)


if __name__ == "__main__":
    main()
