.PHONY: setup format lint type test audit features train backtest ablation deadcode purge-cache paper-trade scheduler

setup:
	python -m pip install --upgrade pip
	python -m pip install -e .[dev]

format:
	ruff format .

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

purge-cache:
	python -m scripts.purge_feature_cache $(ARGS)

paper-trade:
	PYTHONPATH=src python -m scripts.paper_trade $(ARGS)

scheduler:
	PYTHONPATH=src python scripts/run_scheduler.py

scheduler-bg:
	nohup env PYTHONPATH=src python scripts/run_scheduler.py > logs/scheduler.log 2>&1 &
