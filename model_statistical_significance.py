import numpy as np
import pandas as pd
from math import erfc, sqrt

def _p_two_tailed_from_z(z):
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    return float(erfc(abs(z) / sqrt(2)))

def summarize_non_asc_prefix(model, param_names=None):
    """
    Return a DataFrame for *non-ASC* coefficients:
    columns: name, coef, se, z, p, stars.

    Rule: betas are parameters whose names do NOT start with 'ASC_'.
    """

    # ---- coefficients ----
    beta = None
    names = None

    # Try common attrs first
    if hasattr(model, "params"):
        params_obj = getattr(model, "params")
        try:
            # pandas Series
            beta = np.asarray(params_obj).ravel()
            if hasattr(params_obj, "index"):
                names = list(params_obj.index)
        except Exception:
            beta = np.asarray(params_obj).ravel()

    if beta is None:
        for attr in ("beta", "coefficients", "est_params"):
            if hasattr(model, attr):
                beta = np.asarray(getattr(model, attr)).ravel()
                break
    if beta is None:
        raise RuntimeError("Could not find coefficient vector on model.")

    # ---- names fallback ----
    if names is None:
        for attr in ("param_names", "names", "column_labels"):
            if hasattr(model, attr) and getattr(model, attr) is not None:
                names = list(getattr(model, attr))
                break
    if names is None and hasattr(model, "design"):
        try:
            names = list(model.design.columns)
        except Exception:
            pass
    if names is None:
        names = list(param_names) if param_names is not None else [f"beta_{i}" for i in range(len(beta))]

    # ---- covariance (prefer provided; else invert Hessian) ----
    cov = None
    for attr in ("covariance", "cov", "vcov", "cov_params_", "robust_covariance", "robust_cov"):
        if hasattr(model, attr):
            cov = np.asarray(getattr(model, attr))
            break
    if cov is None:
        H = None
        for attr in ("hessian", "_hessian", "H"):
            if hasattr(model, attr):
                H = np.asarray(getattr(model, attr))
                break
        if H is None:
            raise RuntimeError("No covariance or Hessian found to compute SEs.")
        try:
            cov = np.linalg.inv(-H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(-H)

    # ---- assemble ----
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z  = np.divide(beta, se, out=np.full_like(beta, np.nan, dtype=float), where=se > 0)
    p  = np.array([_p_two_tailed_from_z(zi) if np.isfinite(zi) else np.nan for zi in z])

    df = pd.DataFrame({"name": names, "coef": beta, "se": se, "z": z, "p": p})

    # ---- keep only non-ASC (name does NOT start with 'ASC_') ----
    non_asc_mask = ~df["name"].astype(str).str.startswith("ASC_")
    df = df.loc[non_asc_mask].copy()

    # ---- stars ----
    def stars(pv):
        if pv < 0.001: return "***"
        if pv < 0.01:  return "**"
        if pv < 0.05:  return "*"
        return ""
    df["stars"] = [stars(pv) for pv in df["p"]]

    return df[["name", "coef", "se", "z", "p", "stars"]].reset_index(drop=True)

""" ========== Usage:

betas = summarize_non_asc_prefix(model)
print(betas)  # your 17 non-ASC coefficients with se, p, and stars
"""