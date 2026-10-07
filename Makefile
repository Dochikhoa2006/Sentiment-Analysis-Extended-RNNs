PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: install install-dev install-hooks check lint format test download prepare embeddings train evaluate predict docker

install:
	"$(PYTHON)" -m pip install -e ".[all]"

install-dev:
	"$(PYTHON)" -m pip install -e ".[all,dev]"

install-hooks:
	@current=$$(git config --get core.hooksPath || true); \
	if [ -n "$$current" ] && [ "$$current" != ".githooks" ]; then \
		printf '%s\n' "Existing hooksPath: $$current. Add 'make lint' to that hook instead."; \
		exit 1; \
	fi
	git config --local core.hooksPath .githooks

check: lint test

lint:
	"$(PYTHON)" -m ruff check src tests
	"$(PYTHON)" -m ruff format --check src tests

format:
	"$(PYTHON)" -m ruff check --fix src tests
	"$(PYTHON)" -m ruff format src tests
	$(MAKE) lint

test:
	"$(PYTHON)" -m pytest --cov=sentiment_analyzer --cov-report=term-missing

download:
	sentiment-analyzer download

prepare:
	sentiment-analyzer prepare

embeddings:
	sentiment-analyzer embeddings

train:
	sentiment-analyzer train

evaluate:
	sentiment-analyzer evaluate --architectures lstm gru

predict:
	sentiment-analyzer predict

docker:
	docker build -t app-review-sentiment .
