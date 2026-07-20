# Appliance lifespan estimation from Peru's ENAHO survey

This repository contains the data, reported estimates, and Python code used to
estimate household-appliance lifespans from Peru's ENAHO survey for 2008–2024.

The repository URL is retained for continuity. The contents and documentation
use neutral names that describe the analysis.

## Repository contents

```text
data/enaho_module_18/                  Compressed ENAHO module 18 files, 2008–2024
results/weibull_mixture_estimates.csv  Reported mixture estimates
results/ownership_penetration_2008_2024.csv
src/reproduce_results.py               Reproduction script
requirements.txt                       Tested Python dependencies
```

Generated files are written to `reproduced/`, which is excluded from version
control.

## Method

For each appliance and year, the code constructs the observed lifetime as the
survey year minus the reported acquisition year. Observations with nonpositive
lifetimes or lifetimes above 80 years are excluded. ENAHO survey weights are
used throughout.

A two-component Weibull mixture is fitted by expectation–maximization. The
component scales are initialized at the weighted 25th and 75th percentiles of
the observed lifetimes. The initial shapes are 1.0 and 1.8, and the initial weight of the longer-lived component is 0.4.

The shorter-lived component is interpreted as second use, while the
longer-lived component is interpreted as first use. The reported second-use
proportion is therefore one minus the internal weight of the longer-lived
component.

Fourteen appliance-year fits that encountered numerical optimization failures
in the original standard run were re-estimated using optimization in
log-parameters with a weighted-Weibull score-equation fallback. The
reproduction code preserves those robust refits and uses a local numerical
fallback when necessary to avoid platform-dependent optimizer failures.

## Reproduction

The analysis was tested with:

```text
Python 3.9.13
NumPy 1.21.5
pandas 1.4.4
SciPy 1.7.3
```

Create an environment and install the tested dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete analysis from the repository root:

```bash
python src/reproduce_results.py
```

The recomputed files and a comparison against the reported estimates will be
written to:

```text
reproduced/weibull_mixture_estimates.csv
reproduced/ownership_penetration_2008_2024.csv
reproduced/comparison_summary.txt
```

Run only one part of the analysis with:

```bash
python src/reproduce_results.py --task fit
python src/reproduce_results.py --task ownership
```

## Output convention

The main results file uses descriptive column names:

- `second_use_weight`: estimated proportion assigned to the shorter-lived
  component.
- `first_use_weight`: estimated proportion assigned to the longer-lived
  component.
- `second_use_mean`: mean of the shorter-lived Weibull component.
- `first_use_mean`: mean of the longer-lived Weibull component.
- `mixture_mean`: weighted mean across the two components.
- `fit_status`: `standard` or `robust_refit`.

One appliance-year has a nearly degenerate Weibull component with a very large
shape estimate. Its component mean and mixture mean are stable, although the
shape parameter itself is numerically weakly identified.

## Data

The input files are the ENAHO module 18 extracts used in the analysis. They are stored as `.csv.gz` files and read directly by pandas. Data
remain subject to the terms and attribution requirements of their original
provider. The repository does not relicense the source survey data.
