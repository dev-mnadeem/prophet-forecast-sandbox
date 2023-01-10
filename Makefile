# Local targets come first; they are what you need day to day and none of them
# require Docker. The docker-* targets are the original upstream entry points.
.PHONY: help venv install test test-slow lint fmt backtest determinism datasets \
        docker-build docker-shell docker-py-shell

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

venv: $(BIN)/python  ## Create the virtualenv

install: venv  ## Install pinned dev deps and the vendored Prophet (compiles CmdStan; slow)
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e python/

test: ## Run the fast first-party suite (no Stan binary needed)
	$(BIN)/python -m pytest playground -q

test-slow: ## Run the tests that really fit Prophet
	$(BIN)/python -m pytest playground -q -m slow

test-vendored: ## Run the upstream Prophet suite (needs a compiled Stan model)
	$(BIN)/python -m pytest python/prophet/tests python_shim -q

lint: ## Lint first-party code only (python/ and python_shim/ are vendored)
	$(BIN)/ruff check playground docker-healthcheck.py
	$(BIN)/ruff format --check playground docker-healthcheck.py

fmt: ## Autoformat first-party code
	$(BIN)/ruff check --fix playground docker-healthcheck.py
	$(BIN)/ruff format playground docker-healthcheck.py

datasets: ## List pinned datasets and verify their checksums
	$(BIN)/python -m playground datasets

backtest: ## Prophet vs baselines on identical walk-forward folds
	$(BIN)/python -m playground backtest --dataset peyton --per-fold

determinism: ## Measure run-to-run variation in Prophet's output
	$(BIN)/python -m playground determinism

docker-build: ## Build the runtime image
	docker compose build

docker-shell: ## Shell into the runtime image
	docker compose run --rm playground bash

docker-py-shell: ## Python REPL inside the runtime image
	docker compose run --rm playground python
