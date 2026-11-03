"""Train the HDB valuation model and evaluate it forward in time.

Target: log(psm / mkt_psm), i.e. price per sqm relative to the lagged market
level, so the model learns relative value and the index carries the trend.
Train on sales before TEST_START, test on everything after (the model is
frozen for ~16 months of test, so this is a conservative error estimate).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data" / "processed", ROOT / "models"
TEST_START = "2025-07-01"
CAL_MONTHS = 12  # conformal calibration window just before the test set

CATEGORICAL = ["town", "flat_type", "flat_model"]
NUMERIC = ["floor_area_sqm", "storey", "lease_commence_date", "lease_age", "remaining_lease",
           "lat", "lon", "mrt_dist", "school_dist", "schools_1km", "cbd_dist", "max_floor_lvl",
           "total_dwelling_units", "commercial", "blk_rel", "blk_rel_n", "town_type_rel"]
PARAMS = dict(learning_rate=0.05, num_leaves=127, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_estimators=1500, verbose=-1)
QUANTILES = {"p10": 0.10, "p90": 0.90}


def prepare(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    for c in categorical:
        df[c] = df[c].astype("category")
    df["y"] = np.log(df.psm / df.mkt_psm)
    return df


def to_price(df: pd.DataFrame, pred_log_rel: np.ndarray) -> np.ndarray:
    return np.exp(pred_log_rel) * df.mkt_psm.to_numpy() * df.floor_area_sqm.to_numpy()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ape = np.abs(p - y) / y
    return {"MAPE": float(ape.mean()), "MdAPE": float(np.median(ape)),
            "MAE": float(np.abs(p - y).mean()), "within_5pct": float((ape < 0.05).mean()),
            "within_10pct": float((ape < 0.10).mean()),
            "R2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def fit(train: pd.DataFrame, features: list[str], categorical: list[str], **extra) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**{**PARAMS, **extra})
    m.fit(train[features], train.y, categorical_feature=categorical)
    return m


def conformal_offset(fit_df: pd.DataFrame, cal_df: pd.DataFrame, features: list[str],
                     categorical: list[str]) -> float:
    """CQR: how much to widen the p10/p90 band (in log space) for 80% coverage."""
    lo = fit(fit_df, features, categorical, objective="quantile", alpha=QUANTILES["p10"], n_estimators=600)
    hi = fit(fit_df, features, categorical, objective="quantile", alpha=QUANTILES["p90"], n_estimators=600)
    score = np.maximum(lo.predict(cal_df[features]) - cal_df.y, cal_df.y - hi.predict(cal_df[features]))
    return float(np.quantile(score, 0.8))


def train_and_save(df: pd.DataFrame, prefix: str, categorical: list[str], numeric: list[str],
                   test_start: str, baseline_rel: list[str], breakdown: str,
                   increasing: tuple[str, ...] = ()) -> dict:
    """Forward-in-time backtest, calibrated intervals, then refit on all data and
    save models/{prefix}_{point,p10,p90}.txt, {prefix}_serving.json, {prefix}_eval_report.json.
    baseline_rel: lagged relative-$/sqm columns, first non-null is the comps baseline.
    increasing: features the point estimate may only rise with (LightGBM monotone
    constraint; LightGBM doesn't support it for the quantile models)."""
    features = categorical + numeric
    mono = {"monotone_constraints": [int(f in increasing) for f in features]} if increasing else {}
    df = prepare(df, categorical)
    train, test = df[df.date < test_start], df[df.date >= test_start]
    print(f"[{prefix}] train {len(train)} ({train.date.min():%Y-%m}..{train.date.max():%Y-%m}), "
          f"test {len(test)} ({test.date.min():%Y-%m}..{test.date.max():%Y-%m})")
    y = test.price.to_numpy()
    report = {}

    rel = test[baseline_rel].bfill(axis=1).iloc[:, 0].fillna(1.0)
    report["baseline_comps"] = metrics(y, to_price(test, np.log(rel.to_numpy())))

    model = fit(train, features, categorical, **mono)
    pred = to_price(test, model.predict(test[features]))
    report["lightgbm"] = metrics(y, pred)

    # Prediction interval: quantile models, widened by conformal calibration (CQR).
    cal_start = pd.Timestamp(test_start) - pd.DateOffset(months=CAL_MONTHS)
    offset = conformal_offset(train[train.date < cal_start], train[train.date >= cal_start],
                              features, categorical)
    qmodels = {k: fit(train, features, categorical, objective="quantile", alpha=a, n_estimators=600)
               for k, a in QUANTILES.items()}
    for name, off in [("raw", 0.0), ("calibrated", offset)]:
        lo = to_price(test, qmodels["p10"].predict(test[features]) - off)
        hi = to_price(test, qmodels["p90"].predict(test[features]) + off)
        report[f"interval_p10_p90_{name}"] = {
            "coverage": float(((y >= lo) & (y <= hi)).mean()),
            "median_width_pct": float(np.median((hi - lo) / pred))}
    report["conformal_offset_log"] = offset

    # Error breakdowns.
    t = test.assign(ape=np.abs(pred - y) / y)
    report[f"mape_by_{breakdown}"] = t.groupby(breakdown, observed=True).ape.mean().round(4).to_dict()
    report["mape_by_test_month"] = t.groupby(t.date.dt.strftime("%Y-%m")).ape.mean().round(4).to_dict()
    imp = pd.Series(model.booster_.feature_importance("gain"), index=features)
    report["feature_importance_gain_pct"] = (imp / imp.sum() * 100).round(1).sort_values(ascending=False).to_dict()

    print(json.dumps(report, indent=2))
    MODELS.mkdir(exist_ok=True)
    (MODELS / f"{prefix}_eval_report.json").write_text(json.dumps(report, indent=2))

    # Refit on all data for serving.
    final = {"point": fit(df, features, categorical, **mono),
             **{k: fit(df, features, categorical, objective="quantile", alpha=a, n_estimators=600)
                for k, a in QUANTILES.items()}}
    for k, m in final.items():
        m.booster_.save_model(MODELS / f"{prefix}_{k}.txt")
    cal = df.date >= df.date.max() - pd.DateOffset(months=CAL_MONTHS)
    (MODELS / f"{prefix}_serving.json").write_text(json.dumps({
        "conformal_offset_log": conformal_offset(df[~cal], df[cal], features, categorical),
        "features": features, "categorical": categorical,
        "categories": {c: list(df[c].cat.categories) for c in categorical}}, indent=2))
    print(f"[{prefix}] saved models to", MODELS)
    return report


def main() -> None:
    df = pd.read_parquet(DATA / "transactions.parquet").rename(columns={"resale_price": "price"})
    train_and_save(df, "hdb", CATEGORICAL, NUMERIC, TEST_START, ["blk_rel", "town_type_rel"], "flat_type")


if __name__ == "__main__":
    main()
