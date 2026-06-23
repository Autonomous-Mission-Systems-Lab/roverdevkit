# Top-level developer-experience targets.
#
# All targets are .PHONY; this Makefile is a typing shortcut, not a
# build system. The canonical build paths are still ``pytest``,
# ``uvicorn``, and ``npm`` invoked directly. Targets here just spell
# out the conventional invocation so a new contributor can boot the
# webapp with one command.
#
# Convention:
#   make webapp-dev      → boot backend on :8000 and frontend on :5173
#   make webapp-backend  → backend only
#   make webapp-frontend → frontend only
#   make webapp-test     → backend pytest + frontend lint + frontend build
#   make webapp-build    → frontend production build only
#   make pareto-fronts   → (re)generate canonical Pareto fronts under reports/
#   make figures         → (re)render every manuscript figure under paper/figures/
#   make optimizer-robustness → run multi-seed NSGA-II robustness sweep
#   make deploy-space    → push current HEAD to the Hugging Face Space (manual)
#
# Override ports with `UVICORN_PORT=8001 make webapp-backend`.
# Override the conda env used by python targets with `CONDA_ENV=other`.

.PHONY: webapp-dev webapp-backend webapp-frontend webapp-test webapp-build pareto-fronts optimizer-robustness architecture-crossover figures deploy-space

UVICORN_PORT ?= 8000
VITE_PORT ?= 5173
CONDA_ENV ?= roverdevkit

# Boot both servers in one command. `trap 'kill 0'` propagates Ctrl+C
# to every backgrounded child so the cleanup story stays sane on
# macOS GNU make 3.81 (Apple's bundled version) without `.ONESHELL`.
webapp-dev:
	@echo ">> backend  → http://localhost:$(UVICORN_PORT)"
	@echo ">> frontend → http://localhost:$(VITE_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	  uvicorn webapp.backend.main:app --reload --port $(UVICORN_PORT) & \
	  (cd webapp/frontend && npm run dev -- --port $(VITE_PORT)) & \
	  wait

webapp-backend:
	uvicorn webapp.backend.main:app --reload --port $(UVICORN_PORT)

webapp-frontend:
	cd webapp/frontend && npm run dev -- --port $(VITE_PORT)

webapp-test:
	pytest webapp/backend/tests -q
	cd webapp/frontend && npm run lint && npm run build

webapp-build:
	cd webapp/frontend && npm run build

# Regenerate the canonical evaluator-driven Pareto fronts that ship with
# the repo. The Pareto Explorer tab in the webapp loads these via
# `/pareto/fronts`, so a fresh clone gets a working explorer without
# anyone running NSGA-II live. Re-run after editing scenario configs.
# Defaults (50 pop × 60 gens, ~4 min
# total for all four scenarios) are tuned for offline use; pass extra
# args via SCRIPT_ARGS.
pareto-fronts:
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/generate_pareto_fronts.py $(SCRIPT_ARGS)

optimizer-robustness:
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/run_optimizer_robustness.py $(SCRIPT_ARGS)

architecture-crossover:
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/run_architecture_obstacle_crossover.py $(SCRIPT_ARGS)

# Re-render every manuscript figure from the committed artifacts under
# reports/ into paper/figures/ (the directory main.tex reads). Each figure
# has a dedicated scripts/make_*_figure.py regenerator (no notebook), so this
# target is the single one-command rebuild of all paper figures. Run
# `make pareto-fronts` first if the fronts changed.
figures:
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_pareto_fronts_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_rediscovery_distance_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_rediscovery_overlay_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_peak_solar_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_terramechanics_experiment_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_terramechanics_sensitivity_figure.py
	conda run -n $(CONDA_ENV) --no-capture-output python scripts/make_architecture_obstacle_crossover_figure.py

# Manually deploy the webapp to the dedicated Hugging Face Space (Docker
# SDK). Pushes the current committed HEAD; it does NOT run on git push.
# Requires HF_SPACE_REMOTE to point at the Space git URL and git-lfs to
# be installed (the ~26 MB surrogate bundle exceeds HF's plain-git limit).
# See scripts/deploy_hf_space.sh for the full contract.
deploy-space:
	bash scripts/deploy_hf_space.sh
