"""Estimate today's value of a typical unit of each flat type in every block,
and export the map data to web/data/.

"Today" = the month after the latest transaction (market level and comps are
lagged, exactly as in training).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from features import lagged_group_psm, market_index

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, WEB = ROOT / "data" / "processed", ROOT / "models", ROOT / "web" / "data"
SIMPLIFY_DEG = 5e-6  # ~0.5 m footprint simplification
RECENT_YEARS = 10  # flat types/sizes are taken from sales in this window
BLOCK_COLS = ["town", "lat", "lon", "lease_commence_date", "mrt_dist", "mrt_station", "school_dist",
              "schools_1km", "cbd_dist", "max_floor_lvl", "total_dwelling_units", "commercial"]


def typical_units(df: pd.DataFrame) -> pd.DataFrame:
    recent = df[df.date >= df.date.max() - pd.DateOffset(years=RECENT_YEARS)]
    keys = ["block", "street_name", "flat_type"]
    units = recent.groupby(keys, observed=True).agg(
        floor_area_sqm=("floor_area_sqm", "median"),
        flat_model=("flat_model", lambda s: s.mode().iat[0]),
        n_sales=("resale_price", "size")).reset_index()
    blocks = recent.groupby(["block", "street_name"]).last()[BLOCK_COLS].reset_index()
    units = units.merge(blocks, on=["block", "street_name"])
    units["storey"] = np.clip(np.round(units.max_floor_lvl.fillna(12) / 2), 2, None)
    return units


def simplify(ring: list) -> list:
    coords = Polygon(ring).simplify(SIMPLIFY_DEG, preserve_topology=True).exterior.coords
    return [[round(x, 5), round(y, 5)] for x, y in coords]


def nowcast_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Model features for a typical unit of each block/flat type, as of next month."""
    now_t = int(df.t.max()) + 1
    now = df.date.max() + pd.DateOffset(months=1)
    u = typical_units(df).assign(t=now_t)
    u["lease_age"] = now.year + (now.month - 1) / 12 - u.lease_commence_date
    u["remaining_lease"] = 99 - u.lease_age
    u["mkt_psm"] = market_index(df).loc[now_t]
    u = u.merge(lagged_group_psm(df, ["block", "street_name"], 24, "blk_rel"),
                on=["block", "street_name", "t"], how="left")
    u = u.merge(lagged_group_psm(df, ["town", "flat_type"], 6, "town_type_rel"),
                on=["town", "flat_type", "t"], how="left")
    return u, now


class Valuer:
    """Loads the trained models once; predicts est/lo/hi for rows of a nowcast frame."""

    def __init__(self, prefix: str = "hdb") -> None:
        serving = json.loads((MODELS / f"{prefix}_serving.json").read_text())
        self.features, self.categories = serving["features"], serving["categories"]
        self.offset = serving["conformal_offset_log"]
        self.models = {k: lgb.Booster(model_file=str(MODELS / f"{prefix}_{k}.txt"))
                       for k in ["point", "p10", "p90"]}

    def predict(self, u: pd.DataFrame) -> pd.DataFrame:
        u = u.copy()
        for c, values in self.categories.items():
            u[c] = pd.Categorical(u[c].astype(str), categories=values)
        X = u[self.features]
        area_mkt = u.mkt_psm * u.floor_area_sqm
        u["est"] = np.exp(self.models["point"].predict(X)) * area_mkt
        u["lo"] = np.exp(self.models["p10"].predict(X) - self.offset) * area_mkt
        u["hi"] = np.exp(self.models["p90"].predict(X) + self.offset) * area_mkt
        # Quantile models are fit independently; keep the band around the point estimate.
        u["lo"] = np.minimum(u.lo, u.est)
        u["hi"] = np.maximum(u.hi, u.est)
        u["psm"] = u.est / u.floor_area_sqm
        return u


def main() -> None:
    df = pd.read_parquet(DATA / "transactions.parquet")
    df = df[df.lat.notna()]
    frame, now = nowcast_frame(df)
    frame.to_parquet(DATA / "nowcast_units.parquet")  # served by server.py for unit queries
    u = Valuer().predict(frame)
    print(f"valuation month {now:%Y-%m}: {len(u)} block/flat-type units, "
          f"{u[['block', 'street_name']].drop_duplicates().shape[0]} blocks")

    footprints = pd.read_parquet(DATA / "blocks.parquet")[["block", "street_name", "footprint"]]
    tx = df[df.date >= df.date.max() - pd.DateOffset(years=5)].sort_values("date", ascending=False)
    tx_by_block = {k: g.head(20)[["month", "flat_type", "storey_range", "floor_area_sqm", "resale_price"]]
                   .astype({"floor_area_sqm": float, "resale_price": int}).values.tolist() for k, g in tx.groupby(["block", "street_name"])}

    out = []
    for (block, street), g in u.groupby(["block", "street_name"]):
        b = g.iloc[0]
        fp = footprints[(footprints.block == block) & (footprints.street_name == street)].footprint
        poly = json.loads(fp.iat[0]) if len(fp) and fp.iat[0] else None
        out.append({
            "id": f"{block}|{street}", "block": block, "street": street, "town": b.town,
            "lat": round(b.lat, 6), "lon": round(b.lon, 6),
            "poly": [simplify(ring) for ring in poly] if poly else None,
            "floors": int(b.max_floor_lvl) if pd.notna(b.max_floor_lvl) else None,
            "lease_start": int(b.lease_commence_date),
            "mrt": b.mrt_station.title().replace(" Mrt Station", "").replace(" Lrt Station", " LRT"),
            "mrt_m": int(b.mrt_dist), "schools_1km": int(b.schools_1km),
            "psm": int(np.average(g.psm, weights=g.n_sales)),
            "units": [{"type": r.flat_type, "sqm": round(r.floor_area_sqm, 1), "est": int(r.est),
                       "lo": int(r.lo), "hi": int(r.hi), "n": int(r.n_sales)}
                      for r in g.sort_values("floor_area_sqm").itertuples()],
            "tx": tx_by_block.get((block, street), []),
        })
    WEB.mkdir(parents=True, exist_ok=True)
    meta = {"valuation_month": f"{now:%Y-%m}", "latest_sale": f"{df.date.max():%Y-%m}",
            "eval": json.loads((MODELS / "hdb_eval_report.json").read_text())["lightgbm"],
            "forecast": json.loads((MODELS / "forecast.json").read_text())}
    (WEB / "blocks.json").write_text(json.dumps({"meta": meta, "blocks": out}, separators=(",", ":")))
    print(f"wrote {len(out)} blocks, {(WEB / 'blocks.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
