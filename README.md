# Parcoursup FGH estimation

[![CI](https://github.com/denisdsokolov/parcoursup-fgh-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/denisdsokolov/parcoursup-fgh-estimation/actions/workflows/ci.yml)

Research code for estimating student preferences and comparing matching mechanisms on the
French Parcoursup higher-education admissions market. The pipeline prepares restricted-access
administrative data, estimates an FGH discrete-choice model, simulates ranked-order lists, and
compares Parcoursup's call-order mechanism (PCS/CDA) with SF-CDA in counterfactual markets.

## What is included

- `FGH_data_proc.py` — transforms raw annual Parcoursup extracts into analysis tables.
- `FGH_main.py` — runs estimation, preference simulation, matching, and final analysis.
- `cda_mechanisms.py` — program-proposing CDA and SF-CDA implementations.
- `panel_builder.py`, `estimate_fgh.py`, `alpha_ols.py` — FGH estimation pipeline.
- `simulate_rols.py`, `pref_profile_sampling.py` — preference-list simulation.
- `results/` — aggregate simulation outputs and residency histograms.
- `legacy/` — earlier monolithic implementations retained for reference.

Raw and processed individual-level data are deliberately excluded from the repository.
See [docs/DATA.md](docs/DATA.md) for the expected files and privacy boundaries.

## Requirements

- Python 3.10–3.12 (Python 3.12 is used in CI)
- Enough memory for the full administrative extracts; the annual market contains roughly
  880,000 candidates and 21,000 programs.

Create an isolated environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An existing Conda environment with the same Python version can be used instead.

## Configuration

Copy `.env.example` as a reference and export the variables you want to override. The scripts
do not load `.env` automatically, so configuration remains explicit in the shell:

```bash
export PARCOURSUP_YEAR=24
export PARCOURSUP_DATA_DIR=FGH_data_Yc_Yq
export PARCOURSUP_BUNCH_SIZE=10
export PARCOURSUP_BUNCHES=50
export PARCOURSUP_SEED=2024
```

Defaults match the current 2024 experiment. Boolean values accept `true`/`false`, `yes`/`no`,
`on`/`off`, or `1`/`0`.

## Running the pipeline

Place the restricted raw inputs outside version control, then preprocess them:

```bash
python FGH_data_proc.py
```

After the six processed tables and the required coefficient files are available in
`PARCOURSUP_DATA_DIR`, run the simulations:

```bash
python FGH_main.py
```

For a small smoke run, reduce the number of markets without editing the source:

```bash
PARCOURSUP_BUNCH_SIZE=1 PARCOURSUP_BUNCHES=1 python FGH_main.py
```

The simulation seed is derived from `PARCOURSUP_SEED` and the market index, so repeated runs
with the same inputs and configuration reproduce the same preference draws.

## Tests

The unit tests use synthetic data and do not require Parcoursup extracts:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

GitHub Actions compiles the source tree and runs these tests on every push and pull request.

## Research outputs

Aggregate results are stored in `results/simulations/`; residency matrices and figures are in
`results/resident_histograms/`. Their data dictionaries are kept next to the files.

## License

No open-source license has been selected. The code remains under the author's copyright unless
a license is added later; the Parcoursup source data are governed separately by their access
terms.
