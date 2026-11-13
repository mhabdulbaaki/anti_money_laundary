# Finding Money Laundering in Bank Transfers

DSCI 6601 (Practical Machine Learning, Fall 2026) course project: explainable risk scores and
AI-drafted reports on IBM's synthetic AMLworld data (HI-Small).

The project answers one daily question: **which transactions and accounts should an analyst review
first, and why?** It (1) scores every transfer, (2) rolls the scores up into a daily ranked list of
accounts, and (3) explains each alert and has an AI assistant draft a suspicious transaction report
that is checked automatically.

## Results (test days, used once)

| | F1 | Precision | Recall |
|---|---|---|---|
| Warning-sign rules (tuned on validation) | 0.001 | 0.001 | 0.002 |
| XGBoost, normal days (Sep 8 16:12 – Sep 10) | 0.499 | 0.85 | 0.35 |
| XGBoost, including the tail days (Sep 11 – 18) | 0.646 | 0.91 | 0.50 |

- Daily account review list: 95% of each test day's top 100 accounts are truly involved (rules: 10%).
- Without payment format and currency (simulator shortcuts), validation F1 drops from 0.60 to 0.50.
- Explanations: the top-3 TreeSHAP reasons match the true scheme's warning-sign family for 55% of caught
  laundering, vs 43% for shuffled schemes (p < 0.001) but 77% for three random features.
- The Claude drafting step has not been run yet (no API key on the build machine); the template drafter passes
  the format and number checks for all 30 cases.

Logistic regression trains on every laundering row plus 10% of normal rows (saga for L1), because the full
L1 fit did not finish in an hour; LightGBM uses `min_child_weight=1` to stay stable on 1:1000 data.

## Data

Download `HI-Small_Trans.csv`, `HI-Small_Patterns.txt` and `HI-Small_accounts.csv` from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data)
into `data/`. The licence (CDLA-Sharing-1.0) and the file sizes mean the data is **not** in this repository.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python scripts/01_prepare.py          # quality checks, US$ conversion, features (~1 min)
.venv/bin/python scripts/02_cv.py               # time-ordered CV of all candidates (~1-2 h on 8 cores)
.venv/bin/python scripts/03_eda.py              # exploratory charts -> figures/eda_*.png
.venv/bin/python scripts/04_final.py            # select, tune threshold on validation, test once
.venv/bin/python scripts/05_explain_accounts.py # TreeSHAP reasons vs true scheme; daily top-100 accounts
.venv/bin/python scripts/06_draft_reports.py    # evidence packets + STR drafts + automatic checks
.venv/bin/python scripts/07_build_report.py     # fills the course template -> report/
```

`06_draft_reports.py` uses Claude (`claude-opus-5-5`) when `ANTHROPIC_API_KEY` is set, and a
deterministic template otherwise (`--drafter llm|template` to force one). The request enables
server-side refusal fallbacks (`fallbacks: "default"`), so a declined request is retried on a
fallback model.

The exploration notebook `notebooks/01_eda.ipynb` walks through every chart with the decision it led to.

Tests and type checks:

```bash
.venv/bin/pytest -q
.venv/bin/mypy src scripts
```

## Layout

| Path | What it holds |
|---|---|
| `src/aml/data.py` | reading the three files; exact-duplicate removal; scheme typology join |
| `src/aml/fx.py` | fixed exchange rates recovered from currency-changing transfers; US$ conversion |
| `src/aml/features.py` | features from **earlier** transfers only: rolling 1 h / 24 h / 7 d sender and receiver activity, pass-through ratios, loop / path / first-payment flags |
| `src/aml/split.py` | 60/20/20 time split and expanding-window CV folds, cut at minute boundaries |
| `src/aml/rules.py` | warning-sign rule baseline (fan-out, fan-in, pass-through, just below US$10,000) |
| `src/aml/models.py` | scikit-learn Pipelines for logistic regression, LightGBM, XGBoost; grids; CV runner; TreeSHAP |
| `src/aml/metrics.py` | F1 / PR-AUC / best threshold / daily precision of the top-k |
| `src/aml/accounts.py` | account-day roll-up of transfer scores |
| `src/aml/explain.py` | do the top reasons match the true scheme? (permutation and random baselines) |
| `src/aml/str_agent.py` | evidence-only STR drafting (Claude or template) and the format / number checks |
| `src/aml/eda.py`, `viz.py`, `graph.py` | exploration tables and charts; PageRank |
| `results/` | small JSON/CSV results the report quotes |
| `figures/` | every chart |
| `report/` | the filled course report |

Random seed 42 everywhere. Python 3.12; package versions in `requirements.txt`.
