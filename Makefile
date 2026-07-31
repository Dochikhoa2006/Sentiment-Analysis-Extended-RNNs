.PHONY: install install-dev lint format test download prepare embeddings train evaluate predict docker

install:
	python -m pip install -e ".[all]"

install-dev:
	python -m pip install -e ".[all,dev]"

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest --cov=sentiment_analyzer --cov-report=term-missing

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
