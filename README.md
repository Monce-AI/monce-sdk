[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Rules-10/10_passed-brightgreen.svg?logo=githubactions&logoColor=white)](#rule-discovery-test-suite)
[![Speed](https://img.shields.io/badge/Constructor-<200ms-ff6600.svg?logo=zap&logoColor=white)](#how-it-works)
[![Built on](https://img.shields.io/badge/Built_on-algorithmeai--snake-blueviolet.svg)](https://github.com/Monce-AI/algorithmeai-snake)

<p align="center">
  <img src="assets/architecture.svg" width="680" alt="Monce Architecture"/>
</p>

# Monce SDK

**Data to insights. One line. Zero config.**

Turn any DataFrame into a queryable intelligence layer. Predict, regress, explain, discover rules, detect anomalies, fill gaps — all from a single constructor call.

```python
from monce import Oracle
import pandas as pd

oracle = Oracle(pd.read_csv("data.csv"))
print(oracle.context())
```

That's it. Oracle trains a Snake model on every column in parallel, then answers any question about your data instantly.

---

## Why Monce

| Problem | Monce |
|---------|-------|
| "What drives this column?" | `oracle.formula(col="churn")` |
| "Predict this row" | `oracle.predict("churn", row)` |
| "Give my LLM context about this data" | `oracle.context()` |
| "Is this row an outlier?" | `oracle.anomaly_score(row)` |
| "What's the price distribution?" | `oracle.candle("price", row)` |
| "How predictable is each column?" | `oracle.score()` |

No feature engineering. No target selection upfront. No hyperparameter tuning. Data in, intelligence out.

---

## Install

```bash
git clone https://github.com/Monce-AI/monce-sdk.git && cd monce-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e . pandas
python example.py
```

Or: `./setup.sh`

Snake ([algorithmeai](https://github.com/Monce-AI/algorithmeai-snake)) is bundled — zero external dependencies beyond pandas.

---

## Quick Start

```python
import pandas as pd
from monce import Oracle

df = pd.read_csv("titanic.csv")
oracle = Oracle(df, columns=["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"])

# Predict
oracle.predict("Survived", {"Pclass": "1", "Sex": "female", "Age": "17", "Fare": "110", "Embarked": "S"})
# -> 1 (survived), probability: {1: 1.0, 0: 0.0}

# Regress
oracle.regression("Fare", {"Pclass": "3", "Sex": "male", "Age": "20", "Survived": "0", "Embarked": "S"})
# -> 7.65

# Discover rules
print(oracle.formula(col="Survived"))

# Full context for an LLM
print(oracle.context())

# Score all columns
print(oracle.score())
```

---

## Constructor

```python
Oracle(
    data,                # DataFrame or list[dict]
    noise=0.25,          # Snake noise parameter
    workers=1,           # parallel workers per model
    budget=None,         # max training tiers (1-4, None=all)
    n_layers=None,       # override: fixed layer count
    bucket=None,         # override: fixed bucket size
    columns=None,        # subset of columns to model
    target=None,         # default target for predict/regression
)
```

**Quick mode:** `Oracle(data, n_layers=5, bucket=50)` — single tier, fast, good for testing.

**Full mode:** `Oracle(data)` — progressive 4-tier training (10/20/40/80 layers), answers improve over time.

**Focused:** `Oracle(data, target="Survived", columns=["Survived", "Sex", "Age", "Pclass"])` — model only what matters.

---

## API Reference

### Prediction

| Method | Returns | Use case |
|--------|---------|----------|
| `predict(col, row)` | Target value | Classification |
| `probability(col, row)` | `{class: float}` | Confidence scores |
| `regression(col, row)` | `float` | Continuous targets |
| `candle(col, row)` | `Candle` object | Full distribution (high/q3/median/q1/low/mean/std) |
| `audit(col, row)` | Multi-line string | Human-readable reasoning trace |
| `augmented(col, row)` | `dict` | All-in-one: prediction + probability + audit |

If `target=` is set in constructor, `col` can be omitted: `oracle.predict(row)`.

### Discovery

| Method | Returns | Use case |
|--------|---------|----------|
| `formula(col=None)` | Markdown table | Top rules ranked by lift x significance |
| `formula(row, col=None)` | Rule list | Rules that fire for one row |
| `formula(df, col=None)` | Markdown table | Rules across a dataset |
| `score(col=None)` | `dict` | Accuracy / R² per column |
| `correlations()` | `[(col, score, type), ...]` | Column predictability ranking |

### Context

| Method | Returns | Use case |
|--------|---------|----------|
| `context(col=None)` | Markdown string | LLM-ready: columns, stats, rules, summary |
| `anomaly_score(row)` | `{per_column: {...}, overall: float}` | Outlier detection |
| `fill(row, col)` | Predicted value | Missing value imputation |
| `ask(question)` | `dict` | Natural language routing |
| `lookalikes(col, row)` | `[[idx, target, condition], ...]` | Similar training samples |

---

## Context Provider

The killer feature for LLM pipelines. Feed any dataset to Oracle, get back a token-efficient markdown snippet:

```python
print(oracle.context())
```

```markdown
## Dataset Context
**712 rows x 8 columns**

### Columns
- **Survived** (classification) — 2 unique: 0(424), 1(288)
- **Sex** (classification) — 2 unique: male(453), female(259)
- **Age** (regression) — range [0.42, 80.0], mean=29.6
- **Fare** (regression) — range [0.0, 512.33], mean=34.6

### Predictability
- **Pclass**: accuracy=92.3%
- **Survived**: accuracy=80.1%
- **Fare**: R²=0.868

### Discovered Rules
| # | Formula | Lift | Evidence | Sig |
|---|---------|------|----------|-----|
| 1 | IF "Sex" contains "f" AND "Pclass" <= 2 -> Survived = 1 | 2.0x | acc=100%, n=20/712 | ** |

### Summary
This dataset has 712 records across 8 fields. 'Pclass' is predictable at 92%.
'Survived' is predictable at 80%. 'Sex' is binary: male (453/712), female (259/712).
```

One call. Under 2000 tokens. Your LLM now understands the data's structure AND the rules governing it.

---

## How It Works

```
DataFrame (N columns)
    |
    |---> Snake(target="col_1") --> ready
    |---> Snake(target="col_2") --> ready
    |---> Snake(target="col_3") --> ready
    |         ...
    +---> Snake(target="col_N") --> ready

oracle.context()  --> columns + stats + rules + summary
oracle.predict()  --> routes to the right model --> answer + confidence
oracle.formula()  --> top rules ranked by lift x p-value
```

Progressive training: tier 1 trains on 100 rows with 10 layers (fast), then silently upgrades through 20/40/80 layers with more data. Same object, better answers over time.

---

## Rule Discovery

Oracle discovers IF-THEN rules ranked by **lift** (how much better than random) and **statistical significance**.

```
| # | Formula                                          | Lift | Evidence           | Sig |
|---|--------------------------------------------------|------|--------------------|-----|
| 1 | IF "Sex" contains "f" AND "Pclass" <= 2 -> S=1  | 2.0x | acc=100%, n=20/712 | **  |
| 2 | IF "Fare" <= 7.85 AND "Pclass" <= 2 -> S=0      | 2.0x | acc=100%, n=16/712 | **  |
```

- **Classification**: `IF conditions -> target = value` with accuracy, lift, coverage
- **Regression**: `IF conditions -> target ~ median (IQR [q1, q3])` with variance reduction lift

---

## Test Suite

10 synthetic datasets with known ground-truth rules. Oracle must discover each rule from data alone.

| # | Difficulty | Rule | Result |
|---|-----------|------|--------|
| 1 | Easy | `color=red -> yes` | PASS |
| 2 | Easy | `age > 30 -> senior` | PASS |
| 3 | Easy | `name contains "pro" -> premium` | PASS |
| 4 | Medium | `size=big AND color=red -> A` | PASS |
| 5 | Medium | `temp<10->cold, 10-25->mild, >25->hot` | PASS |
| 6 | Medium | `status=active -> yes` (3 noise features) | PASS |
| 7 | Hard | `size=large AND material=wood -> expensive` | PASS |
| 8 | Hard | `department=engineering -> high` (80% noisy) | PASS |
| 9 | Hard | `continent in {NA,EU,AS} -> drives_right` | PASS |
| 10 | Hard | `sqft -> price` (linear regression) | PASS |

Run: `python test_rules.py`

---

## Philosophy

> At start you're an idiot but clever fast insights. At last you're SOTA.

Oracle trains all models in parallel. Progressive intelligence — first models that finish start answering immediately. No waiting for perfection.

**Zero config.** No sklearn. No preprocessing. No feature engineering. Just data in, intelligence out.

**LLM-native.** `.context()` was built for the age of language models. Your data gets a voice.

---

## License

Proprietary — Monce SAS, Paris (SIREN 934 817 198).

View and evaluate freely. Commercial use requires written authorization. See [LICENSE](LICENSE).

---

**Monce SAS** — Paris, France
Built on [algorithmeai-snake](https://github.com/Monce-AI/algorithmeai-snake)
