.RECIPEPREFIX := >
.PHONY: install format lint typecheck test demo

install:
>pip install -e .[dev]

format:
>black .
>isort .

lint:
>ruff check .

typecheck:
>mypy .

test:
>PYTHONPATH=. pytest -q

# simple demo entrypoint
demo:
>PYTHONPATH=src python -m crypto_analyzer.cli
