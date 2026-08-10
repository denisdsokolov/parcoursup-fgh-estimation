import numpy as np
import pandas as pd

def sample_random_profile(rols: dict,
                          students: pd.DataFrame,
                          seed: int | None = None) -> pd.DataFrame:
    if not rols:
        raise ValueError("rols must contain at least one simulated draw")

    rng = np.random.default_rng(seed)
    draw_keys = list(rols.keys())
    n_students = len(students)
    invalid_draws = [key for key, draw in rols.items() if len(draw) != n_students]
    if invalid_draws:
        raise ValueError(
            "Each ROL draw must contain one list per student; "
            f"invalid draws: {invalid_draws}"
        )

    # choose a random draw for each student (uniform)
    draw_indices = rng.integers(low=0, high=len(draw_keys), size=n_students)
    chosen_draws = [draw_keys[j] for j in draw_indices]

    # build long table
    sid = students["student_id"].to_numpy()
    rows = []
    for i, k in enumerate(chosen_draws):
        lst = rols[k][i]
        if not lst:
            continue
        # drop outside (0); keep order
        progs = [p for p in lst if p != 0]
        # if empty after dropping 0, skip (no acceptable prog)
        for rnk, pid in enumerate(progs, start=1):
            rows.append((int(sid[i]), int(pid), int(rnk)))

    prefs = pd.DataFrame(rows, columns=["student_id", "program_id", "rank"])
    return prefs
