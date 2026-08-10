# Data layout and privacy

The repository intentionally excludes the individual-level Parcoursup extracts and all
intermediate pickle files. Candidate identifiers and individual attributes must be handled
according to the access agreement under which the data were obtained.

## Raw inputs

`FGH_data_proc.py` expects the following files for a two-digit year `YY` in its working
directory:

| File | Purpose |
| --- | --- |
| `a_admYY.csv` | Admission proposals and responses |
| `a_recYY.csv` | Courses, targets, and boarding-school capacities |
| `a_rec_grpYY.csv` | Program groups and capacities |
| `a_voeYY.csv` | Candidate wishes |
| `c_can_grpYY.csv` | Merit rankings and call-order rankings |
| `g_canYY.csv` | Candidate attributes |
| `g_tri_insYY.csv` | Institution and program-type attributes |
| `i_insYY.csv` | Program-specific residency status |
| `sp_g_tri_affYY.csv` | Program locations |

Some large tables can be cached locally as `*.pkl`. These caches are ignored by Git.

## Processed inputs

The preprocessing script writes six tables to `PARCOURSUP_DATA_DIR`:

- `data_students.csv`
- `data_programs.csv`
- `data_wishes.csv`
- `data_resident.csv`
- `data_matching.csv`
- `data_ranks.csv`

The estimation stage additionally expects or creates `alpha_hat.csv`, `beta_hat.csv`, and
optionally `rols.pkl.gz` in the same directory.

## Published outputs

Only aggregate simulation tables, aggregate residency histograms, and rendered figures are
kept under `results/`. Before publishing new results, verify that no row can identify an
individual candidate.
