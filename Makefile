.PHONY: setup format lint type test audit features train backtest ablation deadcode

setup:
python -m pip install --upgrade pip
python -m pip install -e .[dev]

format:
black .
isort .

lint:
ruff check .
ruff format --check .

deadcode:
vulture src/crypto_analyzer

type:
mypy .

audit:
deptry .

test:
PYTHONPATH=. pytest -q

features:
python -m scripts.make_features

train:
python -m scripts.train

backtest:
python -m scripts.backtest

ablation:
python -m scripts.ablation
