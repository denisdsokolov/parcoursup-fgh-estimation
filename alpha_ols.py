import numpy as np, pandas as pd


def design_matrix(df):
    """Build the numeric feature matrix used for OLS imputation."""
    X = [np.ones(len(df))]
    X.append(np.log(df["quota"].clip(lower=1)))
    X.append(df["b_share"]); X.append(df["r_share"])
    X.append(df["b_only_type"]); X.append(df["br_type"])
    X.append(df["internat"]); X.append(np.log1p(df["wished_by"]))
    # kind dummies (drop first)
    kinds = pd.get_dummies(df["kind"], prefix = "kind", drop_first=True)
    for col in kinds.columns: X.append(kinds[col])
    return np.column_stack(X)

def impute_alpha(programs: pd.DataFrame, model):
    """Return alpha estimates for every program (sampled MLEs or OLS predictions)."""
    asc = model.params.filter(like="ASC_")
    alpha_samp = pd.DataFrame({"program_id":
                               [int(k.split("_")[1]) for k in asc.index],
                               "alpha_hat": asc.values})
    df = programs.merge(alpha_samp, how = "left") # sampled rows have alphas
    X = design_matrix(df)
    y = df["alpha_hat"].to_numpy()
    mask = ~np.isnan(y) # rows in sample
    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    df["alpha_hat_final"] = np.where(mask, y, X.dot(beta))
    alpha_hat = df[["program_id", "alpha_hat_final"]]

    # adding ASC for outside option 0
    alpha0 = alpha_samp.loc[alpha_samp["program_id"] == 0, "alpha_hat"]
    alpha0_val = float(alpha0.iloc[0])
    alpha_hat = pd.concat(
        [alpha_hat, pd.DataFrame({"program_id": [0], "alpha_hat_final": [alpha0_val]})],
        ignore_index=True
    )
    return alpha_hat
