import pandas as pd
import numpy as np


def sample_programmes(programs: pd.DataFrame, frac: float = 0.30, seed: int | None = 0):
    """Return sampled program IDs, plus ``0`` for the outside option.

    Sampling is stratified by program kind and affirmative-action type. The input
    frame is not modified, and passing the same seed reproduces the same sample.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1]; got {frac}")

    programs = programs.copy()
    rng = np.random.default_rng(seed)
    # stratum = kind x b_only_type x br_type
    programs["stratum"] = (programs["kind"].astype(str) + "_" + programs["b_only_type"].astype(str) + "_" + programs["br_type"].astype(str)
                            )

    print(f"Stratas #: {programs['stratum'].nunique()}")

    keep = []
    for _, g in programs.groupby("stratum"):
        k = max(1, int(len(g)*frac))
        keep.extend(rng.choice(g["program_id"], size = k, replace=False #,p=g["wished_by"]/g["wished_by"].sum()
                               )
                               )

    keep.append(0)
    return set(keep)
