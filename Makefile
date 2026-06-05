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
#
# Override ports with `UVICORN_PORT=8001 make webapp-backend`.
# Override the conda env used by python targets with `CONDA_ENV=other`.

.PHONY: webapp-dev webapp-backend webapp-frontend webapp-test webapp-build pareto-fronts

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
