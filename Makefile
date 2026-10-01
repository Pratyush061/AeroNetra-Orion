.PHONY: install dev test lint demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m orion.cli demo --out outputs

clean:
	rm -rf outputs build dist *.egg-info .pytest_cache .ruff_cache src/*.egg-info
