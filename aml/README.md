# Finding Money Laundering in Bank Transfers

DSCI 6601 (Practical Machine Learning, Fall 2026) course project: explainable risk scores and
AI-drafted reports on IBM's synthetic AMLworld data (HI-Small).

The project answers one daily question: **which transactions and accounts should an analyst review
first, and why?** It

1. **flags** risky transfers with a trained model,
2. **ranks** accounts into a daily review list of the top 100, and
3. **explains** each alert and has an AI assistant **draft a FINTRAC-style report**, which is checked automatically.

## Quick start

Run every command from this `aml/` folder.

```bash
make setup
```

Download `HI-Small_Trans.csv`, `HI-Small_Patterns.txt` and `HI-Small_accounts.csv` from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data)
into `data/raw/`. Then:

```bash
make train
```

```bash
make evaluate
```

`make train` prepares the data and trains the model; `make evaluate` tests it on the test days, explains
the alerts, builds the daily account list and drafts the FINTRAC reports. `make` on its own lists every
command.

| Command | What it does | Runs |
|---|---|---|
| `make setup` | creates `.venv`, installs the packages, creates `ai_agent/.env` | once |
| `make train` | prepares the data (only when it changed), trains the chosen model, sets its alert threshold on the validation days → `model/` | `src/prepare.py`, `src/train.py` |
| `make evaluate` | scores the test days once, explains each alert, ranks accounts, then drafts and checks FINTRAC reports → `outputs/` | `src/evaluate.py`, `ai_agent/draft_reports.py` |
| `make notebook` | runs the notebook top to bottom: exploration, then load and test the model | `notebook/explore_and_test.ipynb` |
| `make select` | compares every candidate model by time-ordered cross-validation (1–2 h); `make train` reuses its results in `outputs/cv/` | `src/select_model.py` |
| `make check` | unit tests and type checks | `tests/` |

## Folders

| Folder | What it holds | In git? |
|---|---|---|
| `data/raw/` | the three Kaggle files | no (licence, size) |
| `data/processed/` | prepared transfers with features, and the scored test days | no |
| `model/` | the trained model, its alert threshold and settings | no (`make train` rebuilds it) |
| `notebook/` | `explore_and_test.ipynb`: Part A explores the data, Part B loads the model and tests it | yes |
| `src/` | the pipeline code | yes |
| `ai_agent/` | the report-drafting assistant and its API-key file `.env` | yes (not `.env`) |
| `outputs/` | metrics, charts and FINTRAC drafts | yes |
| `tests/` | unit tests | yes |

**`src/`**

| File | What it does |
|---|---|
| `config.py` | every path and fixed setting (seed 42, last normal day) |
| `data.py`, `fx.py` | read the files, drop exact duplicates, attach the known schemes; fixed exchange rates → US$ |
| `features.py` | features from **earlier** transfers only: 1 h / 24 h / 7 d activity, pass-through ratios, loop / path / first-payment flags |
| `split.py` | 60/20/20 time split and expanding-window CV folds, cut at minute boundaries |
| `prepare.py` | runs the above → `data/processed/transactions.parquet`, `outputs/data_quality.json` |
| `models.py` | logistic regression, LightGBM and XGBoost as scikit-learn Pipelines; grids; CV runner; TreeSHAP |
| `select_model.py` | cross-validates every candidate → `outputs/cv/` |
| `train.py` | refits the best candidate, tunes its threshold and the rules on validation days, runs the shortcut test → `model/` |
| `rules.py` | the warning-sign rules baseline (fan-out, fan-in, pass-through, just below US$10,000) |
| `metrics.py` | F1, PR-AUC, best threshold, daily precision of the top k |
| `evaluate.py` | scores the test days once and writes the test metrics and Figure 1 |
| `explain.py` | top-3 reasons per alert and whether they match the true scheme |
| `accounts.py` | account-day scores and the daily top-100 review list |
| `eda.py`, `viz.py`, `graph.py` | exploration tables and charts; chart style; PageRank |

**`ai_agent/`** (drafting only: the model that flags transfers lives in `src/`)

| File | What it does |
|---|---|
| `evidence.py` | builds one evidence packet of computed facts per top account: the only thing the AI sees |
| `drafting.py` | the instructions and the three drafters: Gemini, Claude, or a fixed template |
| `checks.py` | checks every draft: the five FINTRAC headings in order, and only numbers from the evidence |
| `draft_reports.py` | drafts and checks the reports for each test day's top 10 accounts → `outputs/fintrac_reports/` |
| `.env` | your API key (git-ignored; `.env.example` is the blank template) |

## The AI drafting key

Put a key in `ai_agent/.env` and `make evaluate` picks the drafter automatically:

- **Gemini** (`gemini-3.8-flash`, free) when `GEMINI_API_KEY` is set. Get a key at
  [Google AI Studio](https://aistudio.google.com/apikey). Free-tier prompts may be used by Google to
  improve its products; the data here is synthetic. Rate-limit errors are retried with backoff.
- **Claude** (`claude-opus-5-5`) when `ANTHROPIC_API_KEY` is set, with server-side refusal fallbacks.
- The fixed **template** when neither is set, so the pipeline always runs end to end.

Every draft, whoever wrote it, goes through the same checks. To force one drafter or model:

```bash
.venv/bin/python -m ai_agent.draft_reports --drafter gemini --model gemini-4-hydrogen
```

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
- AI drafts (Gemini 3.8 Flash, top 10 accounts on each of the 3 normal test days): 30 of 30 well-formed and
  28 of 30 (93%) quote only evidence numbers, just short of the 95% target. The two misses are numbers the
  model worked out itself (a "46 minutes" gap between transfers, and "0 rules" for an empty rule list).

Logistic regression trains on every laundering row plus 10% of normal rows (saga for L1), because the full
L1 fit did not finish in an hour; LightGBM uses `min_child_weight=1` to stay stable on 1:1000 data.

Random seed 42 everywhere. Python 3.12; package versions in `requirements.txt`.
