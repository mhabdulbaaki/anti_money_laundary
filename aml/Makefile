# Finding money laundering in bank transfers: run `make` to list the commands.
# Run every command from this aml/ folder.

PYTHON   ?= python3
PY       := .venv/bin/python
RAW      := data/raw/HI-Small_Trans.csv data/raw/HI-Small_Patterns.txt data/raw/HI-Small_accounts.csv
PREPARED := data/processed/transactions.parquet

.DEFAULT_GOAL := help
.PHONY: help setup train evaluate select notebook check

help:  ## list the commands
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-9s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything (once)
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q -e .
	@test -f ai_agent/.env || cp ai_agent/.env.example ai_agent/.env

train: $(PREPARED)  ## prepare the data (when needed) and train the model -> model/
	$(PY) -m src.train

evaluate:  ## test the model on the test days, then draft and check FINTRAC reports -> outputs/
	$(PY) -m src.evaluate
	$(PY) -m ai_agent.draft_reports

select:  ## compare every candidate model by cross-validation (slow: 1-2 h) -> outputs/cv/
	$(PY) -m src.select_model

notebook:  ## run the notebook top to bottom (exploration, then load and test the model)
	$(PY) -m nbconvert --to notebook --execute --inplace notebook/explore_and_test.ipynb

check:  ## unit tests and type checks
	$(PY) -m pytest -q
	$(PY) -m mypy src ai_agent

# The prepared data is rebuilt only when the raw files or the preparation code change.
$(PREPARED): $(RAW) src/prepare.py src/data.py src/fx.py src/features.py src/split.py
	$(PY) -m src.prepare

$(RAW):
	@echo "Missing $@: download the three HI-Small files from Kaggle into data/raw/ (see data/raw/README.md)"; exit 1
