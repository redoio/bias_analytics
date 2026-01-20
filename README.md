# Introduction
Compute group-level bias metrics from a cohort built from **demographics.csv** (plus optional commitments tables).

This repo avoids hard-coding: you can compare *any* two values in your group column (e.g., any two ethnicities) and define outcomes from *any* demographics column.

## Features

- Build a cohort from demographics (optionally restricted to a list of CDC IDs)
- Optional row filtering via `filters.json`
- Two outcome modes:
  - **Categorical**: outcome = 1 if `outcome_col == outcome_positive`
  - **Numeric threshold**: outcome = 1 if `outcome_col (op) threshold`
- 2×2 contingency table (a,b,c,d)
- Metrics:
  - Odds Ratio (OR) + log CI
  - Relative Risk (RR) + log CI
  - Rate Ratio scaffold (requires person-time inputs; not wired into CLI yet)
- Optional continuity correction (only if explicitly provided)

## Install

From repo root:

```bash
pip install -e .
```

CLI entrypoint:

```bash
bias-analysis --help
```

Or run directly:

```bash
python -m bias_analysis.cli --help
```

## Inputs

### Required

- `--demographics` : path to a CSV (or XLSX if your `read_table()` supports it)

### Optional

- `--current` : current_commitments table (loaded and passed through; used later)
- `--prior` : prior_commitments table (loaded and passed through; used later)
- `--cdc-ids` : optional list of CDC IDs to restrict the cohort

## Filters

Filters are optional. If you don’t want filters, either omit `--filters-file` or use:

`filters.json`
```json
[]
```

Filter objects look like:

```json
[
  {"col": "sex", "op": "eq", "value": "Male"},
  {"col": "facility", "op": "in", "value": ["A", "B", "C"]}
]
```

Supported ops in the current CLI:

- `in` (value must be a list)
- `eq`
- `neq`

Important: if you filter `ethnicity` to a subset (e.g., only `["Black","White"]`), then trying to run `--exposed "American Indian"` will fail because those rows were filtered out. For arbitrary group comparisons, keep filters empty or avoid filtering the group column.

## Outcome modes

### 1. Categorical outcome

Outcome = 1 when:

`demographics[outcome_col] == outcome_positive`

PowerShell example:

```powershell
python -m bias_analysis.cli `
  --demographics demographics.csv `
  --group-col ethnicity `
  --exposed "White" `
  --unexposed "Black" `
  --outcome-col "controlling offense" `
  --outcome-positive "PC187 2nd" `
  --filters-file filters.json `
  --continuity-correction 0.5
```

### 2. Numeric threshold outcome

Outcome = 1 when:

`to_numeric(demographics[outcome_col]) (op) threshold`

Supported `--threshold-op`:

- `ge` (>=)
- `gt` (>)
- `le` (<=)
- `lt` (<)
- `eq` (==)
- `ne` (!=)

Example (sentence length ≥ 10 years):

```powershell
python -m bias_analysis.cli `
  --demographics demographics.csv `
  --group-col ethnicity `
  --exposed "White" `
  --unexposed "Black" `
  --outcome-col "aggregate sentence in years" `
  --outcome-threshold 10 `
  --threshold-op ge
```

## Continuity correction (zero cells)

By default, this repo does not silently “fix” zero cells.

- If any cell in the 2×2 table is zero and you do **not** pass `--continuity-correction`, ratio metrics return **NaN**.
- To enable a continuity correction (common choice: 0.5):

```bash
--continuity-correction 0.5
```

The correction is only applied when a zero cell is present.

## Output

The CLI prints JSON with:

- `inputs` : all inputs used (including filters and whether commitments were loaded)
- `table` : the 2×2 table `{a,b,c,d}`
- `metrics` : computed metrics + CIs



