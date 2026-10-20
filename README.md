# Stable choice with overlapping targets: Parcoursup replication code

[![CI](https://github.com/denisdsokolov/parcoursup-fgh-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/denisdsokolov/parcoursup-fgh-estimation/actions/workflows/ci.yml)

Research code accompanying Denis Sokolov, *Stable Choice with Overlapping Targets:
Theory and Evidence from French College Admissions*. It estimates
applicant preferences from restricted Parcoursup records and compares the current fixed
call-order mechanism (PCS) with Stage-Fair College-Proposing Deferred Acceptance (SF-CDA).

The paper studies a capacity-constrained choice problem with two overlapping traits and
one-to-all accounting. A candidate with both traits counts toward both targets. Stable choices
are indexed by the lowest-priority selected dual-trait candidate (the **anchor**); SF-CDA uses
the highest-priority stable anchor, `k_left`, on the residual newcomer problem at each stage.

## Main empirical result

In 1,000 paired 2024 simulations, SF-CDA matches **9,796 additional scholarship holders**:
100,329 versus 90,533 under PCS, an increase of **10.8%**. Total assignments rise by only
272 (510,994 versus 510,722). Among scholarship holders, 22,037 prefer SF-CDA's assignment
and 2,903 prefer PCS. Justified envy falls from 174,801 candidates to 1,028 on average.
Mean scholarship-target fulfillment rises from 68.21% to 87.84%, measured relative to each
program's **final enrollment**. These are model-based comparisons conditional on estimated
preferences. The manuscript values for 2022–2024 are in
[`results/paper/mechanism_comparison.csv`](results/paper/mechanism_comparison.csv).

SF-CDA is defined by percentage shares of its attainable stage cohort. Non-wastefulness makes
its integer-threshold implementation exactly equivalent stage by stage, including proposals
and rejections. All programs choose before candidates respond. The empirical model treats
boarding programs as regular admission programs. See [`docs/TARGETS.md`](docs/TARGETS.md)
for formulas, rounding, fairness checks, and the scope of the implementation.

Read [`docs/PAPER.md`](docs/PAPER.md) for the paper summary and
[`docs/TRACEABILITY.md`](docs/TRACEABILITY.md) for the claim-by-claim code audit.

## Repository map

- `FGH_data_proc.py` — transforms restricted annual Parcoursup extracts.
- `panel_builder.py`, `estimate_fgh.py`, `alpha_ols.py` — eligibility set, multinomial-logit
  estimation, and alternative-specific-constant imputation.
- `simulate_rols.py`, `pref_profile_sampling.py` — simulated preference profiles.
- `cda_mechanisms.py` — PCS, SF-CDA, final-enrollment target fulfillment, and justified-envy calculations.
- `percentage_targets.py` — integer thresholds induced by declared percentage shares.
- `paper_reporting.py` — paired means and across-simulation standard deviations, as in the paper.
- `FGH_main.py` — end-to-end orchestration; its paper default is 1,000 markets.
- `results/paper/` — manuscript values transcribed from the paper.
- `results/simulations/` — retained development-run aggregates, not exact table replications.
- `legacy/` — superseded scripts retained only for provenance.

Raw and processed individual-level data are deliberately excluded. See
[`docs/DATA.md`](docs/DATA.md) for expected inputs and privacy boundaries.

## Installation

Python 3.10–3.12 is supported (CI uses 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration and execution

The scripts read explicit environment variables; `.env` files are not loaded automatically.
Defaults in `.env.example` reproduce the 2024 paper setup of 10 candidates per bunch and 100
bunches, hence 1,000 paired markets.

```bash
export PARCOURSUP_YEAR=24
export PARCOURSUP_DATA_DIR=FGH_data_Yc_Yq
export PARCOURSUP_BUNCH_SIZE=10
export PARCOURSUP_BUNCHES=100
export PARCOURSUP_SEED=2024

python FGH_data_proc.py
python FGH_main.py
```

For a small one-market run with the required processed inputs:

```bash
PARCOURSUP_BUNCH_SIZE=1 PARCOURSUP_BUNCHES=1 python FGH_main.py
```

Exact end-to-end replication additionally requires the restricted administrative extracts,
the sampled program IDs, and the estimated `alpha_hat.csv` and `beta_hat.csv`; these cannot be
published here. With identical inputs and seed, each simulated preference draw is shared by
PCS and SF-CDA.

For 2022 or 2023, set `PARCOURSUP_YEAR=22` or `23` and point `PARCOURSUP_DATA_DIR` to that year's
processed tables and original coefficients. Without overrides, the run uses 100 paired markets
for those years; `PARCOURSUP_RESULT_YEAR` and the seed follow the selected year. Explicit bunch
settings in your environment take precedence. The included estimator is the 2024 baseline;
other years use their saved annual coefficients. `PARCOURSUP_USE_ROLS=true` uses seeded draws
from the cached profile pool, whereas the default generates fresh profiles.

## Tests

The tests use synthetic data, including the worked anchor example, both PCS counterexamples,
independent exhaustive exchange checks, and complete-history equivalence checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions compiles the source and runs the tests on every push and pull request.

## License

No open-source license has been selected. The code remains under the author's copyright; the
Parcoursup source data are governed separately by their access terms.
