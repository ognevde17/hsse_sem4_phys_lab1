#!/usr/bin/env bash
# Единый скрипт запуска.
# Использование:
#   ./run.sh install
#   ./run.sh plates    [--L 1.0] [--d 0.2] [--nx 12] [--ny 12] [--V 1.0] [--output out.png]
#   ./run.sh spheres   [--R_inner 0.3] [--R_outer 0.6] [--n_theta 20] [--n_phi 40] [--V 1.0] [--output out.png]
#   ./run.sh tests     [-v]
#   ./run.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src/main.py"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<EOF
Использование: $0 <команда> [параметры]

Команды:
  install            установить зависимости из requirements.txt
  plates  [...]      две параллельные пластины
  spheres [...]      две концентрические сферы
  tests   [...]      запустить аналитические тесты (передаётся в pytest)
  server  [PORT]     запустить веб-интерфейс (по умолчанию порт 5000)
  all                запустить plates + spheres + tests с параметрами по умолчанию

Параметры plates (все необязательны, показаны значения по умолчанию):
  --L 1.0            сторона пластины, м
  --d 0.2            расстояние между пластинами, м
  --nx 12            панелей по x на пластину
  --ny 12            панелей по y на пластину
  --V  1.0           разность потенциалов, В
  --output plates.png файл для сохранения графика

Параметры spheres:
  --R_inner 0.3      радиус внутренней сферы, м
  --R_outer 0.6      радиус внешней сферы, м
  --n_theta 20       широтных полос
  --n_phi   40       долготных полос
  --V       1.0      разность потенциалов, В
  --output  spheres.png

Примеры:
  $0 plates
  $0 plates --d 0.05 --nx 15 --ny 15 --output thin.png
  $0 spheres --R_inner 0.5 --R_outer 1.0
  $0 tests -v
  $0 all
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    install)
        $PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"
        ;;
    plates)
        $PYTHON "$SRC" plates "$@"
        ;;
    spheres)
        $PYTHON "$SRC" spheres "$@"
        ;;
    tests)
        $PYTHON -m pytest "$SCRIPT_DIR/tests/test_analytical.py" "$@"
        ;;
    server)
        PORT="${1:-5001}"
        echo "Запуск веб-интерфейса на http://localhost:$PORT"
        PORT=$PORT $PYTHON "$SCRIPT_DIR/web/app.py"
        ;;
    all)
        echo "=== Параллельные пластины ==="
        $PYTHON "$SRC" plates --L 1.0 --d 0.2 --nx 12 --ny 12 --V 1.0 --output plates.png
        echo ""
        echo "=== Концентрические сферы ==="
        $PYTHON "$SRC" spheres --R_inner 0.3 --R_outer 0.6 --n_theta 20 --n_phi 40 --V 1.0 --output spheres.png
        echo ""
        echo "=== Тесты ==="
        $PYTHON -m pytest "$SCRIPT_DIR/tests/test_analytical.py" -v
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Неизвестная команда: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
