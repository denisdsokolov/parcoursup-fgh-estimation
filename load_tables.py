import pandas as pd
from pathlib import Path


REQUIRED_TABLES = (
    "data_students.csv",
    "data_programs.csv",
    "data_wishes.csv",
    "data_resident.csv",
    "data_matching.csv",
    "data_ranks.csv",
)


def load_all(datadir: str | Path):
    """Load and return the six processed Parcoursup analysis tables."""
    p = Path(datadir)
    missing = [name for name in REQUIRED_TABLES if not (p / name).is_file()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(f"Missing processed tables in {p}: {formatted}")

    students = pd.read_csv(p/"data_students.csv", low_memory=False)
    students = students.drop(columns=["post_code"], errors="ignore").drop_duplicates()

    programs = pd.read_csv(p/"data_programs.csv", low_memory=False)
    wishes = pd.read_csv(p/"data_wishes.csv", low_memory=False)
    resident = pd.read_csv(p/"data_resident.csv", low_memory=False)
    matches = pd.read_csv(p/"data_matching.csv", low_memory=False).rename(
        columns={"program_id": "college_id_final"}
    )
    ranks = pd.read_csv(p/"data_ranks.csv", low_memory=False)
    return students, programs, wishes, resident, matches, ranks
