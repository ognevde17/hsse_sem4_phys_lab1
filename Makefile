PYTHON  = python3
SRC     = src/main.py
PYTEST  = $(PYTHON) -m pytest

.PHONY: install plates spheres thin-plates tests server all clean help

help:
	@echo "Доступные цели:"
	@echo "  make install       — установить зависимости"
	@echo "  make server        — веб-интерфейс на http://localhost:5000"
	@echo "  make plates        — пластины (L=1 м, d=0.2 м, 12×12)"
	@echo "  make thin-plates   — тонкие пластины (L=1 м, d=0.05 м, хорошо виден краевой эффект)"
	@echo "  make spheres       — концентрические сферы (R1=0.3 м, R2=0.6 м)"
	@echo "  make tests         — аналитические тесты"
	@echo "  make all           — plates + spheres + tests"
	@echo "  make clean         — удалить PNG-файлы и кэши"

install:
	$(PYTHON) -m pip install -r requirements.txt

server:
	$(PYTHON) web/app.py  # открой http://localhost:5001

plates:
	$(PYTHON) $(SRC) plates \
		--L 1.0 --d 0.2 --nx 12 --ny 12 \
		--V 1.0 --output plates.png

thin-plates:
	$(PYTHON) $(SRC) plates \
		--L 1.0 --d 0.05 --nx 15 --ny 15 \
		--V 1.0 --output plates_thin.png

spheres:
	$(PYTHON) $(SRC) spheres \
		--R_inner 0.3 --R_outer 0.6 \
		--n_theta 20 --n_phi 40 \
		--V 1.0 --output spheres.png

tests:
	$(PYTEST) tests/test_analytical.py -v

all: plates spheres tests

clean:
	rm -f *.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
