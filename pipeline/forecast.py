"""Lease-decay curve and market scenarios for forward projections.

Decay: log(psm / market psm) = block fixed effect + g(lease_age) + controls,
fit within blocks (each block is compared only with itself as it ages), so
location and build quality cancel out. g is a piecewise-linear spline.
Exported as relative value r(age) = exp(g), with r(0) = 1. A projection k years
ahead is: today's estimate x r(age + k) / r(age) x market growth. Beyond the
oldest observed sales there is no data; r falls in a straight line to 0 at 99
years, when the lease returns to the state uncompensated.

Backtest: fit on sales before 2019, then predict each block/flat type's change
in relative value from 2017-18 to 2024-26 against what actually happened.

Output: models/forecast.json (curve with bootstrap band, scenarios, backtest).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data" / "processed", ROOT / "models"
KNOTS = np.arange(5, 60, 5)
MAX_AGE = 99
N_BOOT = 50
MONOTONE = True  # chosen by backtest; see models/forecast.json


def age_basis(age: np.ndarray) -> np.ndarray:
    age = np.asarray(age, dtype=float)
    return np.column_stack([age] + [np.maximum(age - k, 0) for k in KNOTS])


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    controls = pd.get_dummies(df.flat_type, drop_first=True, dtype=float)
    controls["log_area"] = np.log(df.floor_area_sqm)
    controls["storey"] = df.storey
    X = np.column_stack([age_basis(df.lease_age), controls.to_numpy()])
    y = np.log(df.psm / df.mkt_psm).to_numpy()
    return X, y, len(KNOTS) + 1


def per_block_moments(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Within-block demeaned X'X and X'y per block, for fast cluster bootstrap."""
    X, y, _ = design(df)
    g = df.groupby(["block", "street_name"]).ngroup().to_numpy()
    n = g.max() + 1
    counts = np.bincount(g, minlength=n)[:, None]
    Xm = np.zeros((n, X.shape[1]))
    np.add.at(Xm, g, X)
    ym = np.bincount(g, weights=y, minlength=n)
    Xd, yd = X - (Xm / counts)[g], y - (ym / counts[:, 0])[g]
    XtX = np.zeros((n, X.shape[1], X.shape[1]))
    np.add.at(XtX, g, Xd[:, :, None] * Xd[:, None, :])
    Xty = np.zeros((n, X.shape[1]))
    np.add.at(Xty, g, Xd * yd[:, None])
    return XtX, Xty


def solve(XtX: np.ndarray, Xty: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(XtX)) if weights is None else weights
    A = np.einsum("b,bij->ij", w, XtX) + 1e-6 * np.eye(XtX.shape[1])
    return np.linalg.solve(A, np.einsum("b,bi->i", w, Xty))


def curve(beta: np.ndarray, n_age: int, max_observed: int, monotone: bool) -> np.ndarray:
    """Relative value r(age) for integer ages 0..MAX_AGE, r(0) = 1."""
    ages = np.arange(MAX_AGE + 1)
    g = age_basis(ages) @ beta[:n_age]
    g -= g[0]
    if monotone:
        seen = ages <= max_observed
        g[seen] = IsotonicRegression(increasing=False).fit_transform(ages[seen], g[seen])
    r = np.exp(g)
    tail = ages > max_observed
    r[tail] = r[max_observed] * (MAX_AGE - ages[tail]) / (MAX_AGE - max_observed)
    return r


def fit(df: pd.DataFrame, boot: int = 0, seed: int = 0, monotone: bool = MONOTONE) -> dict:
    XtX, Xty = per_block_moments(df)
    n_age = len(KNOTS) + 1
    max_obs = int(np.floor(df.lease_age.quantile(0.999)))
    beta = solve(XtX, Xty)
    out = {"r": curve(beta, n_age, max_obs, monotone), "max_observed_age": max_obs}
    if boot:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(boot):
            w = np.bincount(rng.integers(0, len(XtX), len(XtX)), minlength=len(XtX)).astype(float)
            draws.append(curve(solve(XtX, Xty, w), n_age, max_obs, monotone))
        out["r_lo"], out["r_hi"] = np.percentile(draws, [5, 95], axis=0)
    return out


def backtest(df: pd.DataFrame, monotone: bool = MONOTONE) -> dict:
    train = df[df.date < "2019-01-01"]
    g = np.log(np.maximum(fit(train, monotone=monotone)["r"], 1e-6))
    keys = ["block", "street_name", "flat_type"]
    rel = np.log(df.psm / df.mkt_psm)
    a = df[(df.date >= "2017-01-01") & (df.date < "2019-01-01")].assign(rel=rel).groupby(keys)
    b = df[df.date >= "2024-07-01"].assign(rel=rel).groupby(keys)
    a = a.agg(rel_a=("rel", "mean"), age_a=("lease_age", "mean"), n_a=("rel", "size"))
    b = b.agg(rel_b=("rel", "mean"), age_b=("lease_age", "mean"), n_b=("rel", "size"))
    m = a.join(b, how="inner")
    m = m[(m.n_a >= 3) & (m.n_b >= 3)]
    interp = lambda age: np.interp(age, np.arange(MAX_AGE + 1), g)
    m["actual"] = m.rel_b - m.rel_a
    m["pred"] = interp(m.age_b) - interp(m.age_a)
    # Both sides are relative to the market, whose own drift is unknown; compare
    # after removing the average change, i.e. who out/under-performs whom.
    act, pred = m.actual - m.actual.mean(), m.pred - m.pred.mean()
    m["age_bucket"] = pd.cut(m.age_a, [0, 10, 20, 30, 40, 50, 60], right=False)
    by_age = m.assign(act=act, pred=pred).groupby("age_bucket", observed=True).agg(
        n=("act", "size"), actual_pct=("act", "mean"), predicted_pct=("pred", "mean"))
    return {
        "pairs": int(len(m)),
        "horizon_years": round(float((m.age_b - m.age_a).mean()), 1),
        "rmse_no_change": float(np.sqrt((act ** 2).mean())),
        "rmse_decay_model": float(np.sqrt(((act - pred) ** 2).mean())),
        "corr_pred_actual": float(np.corrcoef(act, pred)[0, 1]),
        "by_start_age": (by_age * [1, 100, 100]).round(2).reset_index().astype({"age_bucket": str})
                        .to_dict(orient="records"),
    }


def scenarios(df: pd.DataFrame) -> list[dict]:
    yearly = df.groupby(df.date.dt.year).psm.median()
    cagr = lambda a, b: float((yearly[b] / yearly[a]) ** (1 / (b - a)) - 1)
    last = int(df.date.max().year)
    return [
        {"name": "Cooling (like 2013–19)", "rate": cagr(2013, 2019)},
        {"name": "Flat market", "rate": 0.0},
        {"name": f"Long-run (2012–{last % 100})", "rate": cagr(2012, last)},
        {"name": "Boom (like 2020–25)", "rate": cagr(2020, 2025)},
    ]


def main() -> None:
    df = pd.read_parquet(DATA / "transactions.parquet")
    df = df[df.lat.notna()]
    bt = backtest(df)
    alt = backtest(df, monotone=not MONOTONE)
    bt[f"rmse_decay_model_monotone_{not MONOTONE}"] = alt["rmse_decay_model"]
    print(json.dumps(bt, indent=2))
    full = fit(df, boot=N_BOOT)
    out = {
        "decay": {k: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v for k, v in full.items()},
        "scenarios": scenarios(df),
        "backtest": bt,
    }
    (MODELS / "forecast.json").write_text(json.dumps(out, indent=2))
    r = full["r"]
    print("relative value vs new flat (age: %):",
          {a: round(100 * (r[a] - 1), 1) for a in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]})
    print("scenarios:", [(s["name"], round(100 * s["rate"], 2)) for s in out["scenarios"]])


if __name__ == "__main__":
    main()
