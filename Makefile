# Two ways to run everything: a local virtualenv (first block) or the Docker
# image (docker-* targets). Neither compiles anything; Prophet installs as a
# prebuilt wheel.
.PHONY: help venv install test test-slow lint fmt datasets backtest determinism \
        docker-build docker-test docker-backtest docker-shell

PYTHON ?= python3.12
VENV   ?= .venv
BIN    := $(VENV)/bin
COMPOSE ?= docker compose

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

venv: $(BIN)/python  ## Create the virtualenv

install: venv  ## Install the pinned runtime and dev dependencies
	$(BIN)/pip install -r requirements-dev.txt

test: ## Run the fast suite (no Prophet fits)
	$(BIN)/python -m pytest playground -q

test-slow: ## Run the tests that really fit Prophet
	$(BIN)/python -m pytest playground -q -m slow

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

docker-build: ## Build the image
	$(COMPOSE) build playground

docker-test: ## Run both test suites and the linter inside the image
	$(COMPOSE) run --rm playground sh -c 'python -m pytest playground -q -p no:cacheprovider && python -m pytest playground -q -p no:cacheprovider -m slow && ruff check --no-cache playground docker-healthcheck.py'

docker-backtest: ## Run the default backtest inside the image
	$(COMPOSE) run --rm playground

docker-shell: ## Shell into the image
	$(COMPOSE) run --rm playground bash
