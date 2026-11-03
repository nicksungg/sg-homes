"""Build the modelling table: one row per resale transaction with location,
block, lease and lagged-market features. Output: data/processed/transactions.parquet

Every market feature for month m uses only transactions before m, so the table
can be used for honest forward-in-time validation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
EARTH_M = 6_371_000
CBD = (1.2840, 103.8514)  # Raffles Place


def load_resale() -> pd.DataFrame:
    df = pd.concat(pd.read_csv(p, dtype={"block": str}) for p in sorted(RAW.glob("resale_*.csv")))
    df = df.drop(columns=["remaining_lease"])
    df["block"] = df.block.str.strip().str.upper()
    df["street_name"] = df.street_name.str.strip().str.upper()
    df["flat_type"] = df.flat_type.str.replace("MULTI GENERATION", "MULTI-GENERATION")
    df["flat_model"] = df.flat_model.str.upper()
    df["date"] = pd.to_datetime(df.month)
    df["t"] = (df.date.dt.year - 2012) * 12 + df.date.dt.month - 1  # month index
    lo, hi = df.storey_range.str.split(" TO ", expand=True).astype(int).T.values
    df["storey"] = (lo + hi) / 2
    df["lease_age"] = df.date.dt.year + (df.date.dt.month - 1) / 12 - df.lease_commence_date
    df["remaining_lease"] = 99 - df.lease_age
    df["psm"] = df.resale_price / df.floor_area_sqm
    return df.sort_values("date").reset_index(drop=True)


def radians(df: pd.DataFrame) -> np.ndarray:
    return np.deg2rad(df[["lat", "lon"]].to_numpy())


def location_features(blocks: pd.DataFrame) -> pd.DataFrame:
    mrt = pd.read_parquet(OUT / "mrt_exits.parquet")
    schools = pd.read_parquet(OUT / "schools.parquet")
    q = radians(blocks)

    d, i = BallTree(radians(mrt), metric="haversine").query(q, k=1)
    blocks["mrt_dist"] = d[:, 0] * EARTH_M
    blocks["mrt_station"] = mrt.station.to_numpy()[i[:, 0]]

    st = BallTree(radians(schools), metric="haversine")
    blocks["school_dist"] = st.query(q, k=1)[0][:, 0] * EARTH_M
    blocks["schools_1km"] = st.query_radius(q, r=1000 / EARTH_M, count_only=True)

    lat, lon = np.deg2rad(blocks.lat), np.deg2rad(blocks.lon)
    clat, clon = np.deg2rad(CBD[0]), np.deg2rad(CBD[1])
    a = np.sin((lat - clat) / 2) ** 2 + np.cos(lat) * np.cos(clat) * np.sin((lon - clon) / 2) ** 2
    blocks["cbd_dist"] = 2 * EARTH_M * np.arcsin(np.sqrt(a))
    return blocks


def block_info(blocks: pd.DataFrame) -> pd.DataFrame:
    pi = pd.read_csv(RAW / "hdb_property_info.csv", dtype={"blk_no": str})
    pi = pi.rename(columns={"blk_no": "block", "street": "street_name"})
    pi["block"] = pi.block.str.strip().str.upper()
    pi["street_name"] = pi.street_name.str.strip().str.upper()
    pi = pi[["block", "street_name", "max_floor_lvl", "total_dwelling_units", "commercial",
             "market_hawker", "multistorey_carpark"]]
    for c in ["commercial", "market_hawker", "multistorey_carpark"]:
        pi[c] = (pi[c] == "Y").astype(int)
    return blocks.merge(pi, on=["block", "street_name"], how="left")


def market_index(df: pd.DataFrame) -> pd.Series:
    """Lagged market level per month: median $/sqm over the 3 months before m.
    Indexed by month index t; covers t = max(t)+1 too, for nowcasting."""
    monthly = df.groupby("t").psm.median()
    full = monthly.reindex(range(monthly.index.min(), monthly.index.max() + 2))
    return full.rolling(3, min_periods=1).mean().shift(1).rename("mkt_psm")


def lagged_group_psm(df: pd.DataFrame, keys: list[str], months: int, name: str) -> pd.DataFrame:
    """Mean of (psm / mkt_psm) for the same group over the previous `months` months,
    excluding the current month. Returns columns name, name_n."""
    g = (df.assign(rel=df.psm / df.mkt_psm)
           .groupby(keys + ["t"]).rel.agg(["sum", "count"]).reset_index())
    out = []
    for _, grp in g.groupby(keys, sort=False):
        s = grp.set_index("t")[["sum", "count"]]
        s = s.reindex(range(s.index.min(), s.index.max() + months + 1), fill_value=0)
        r = s.rolling(months, min_periods=1).sum().shift(1)
        r[keys] = grp[keys].iloc[0].to_numpy()
        out.append(r.reset_index())
    r = pd.concat(out)
    r[name] = r["sum"] / r["count"].replace(0, np.nan)
    r[f"{name}_n"] = r["count"]
    return r[keys + ["t", name, f"{name}_n"]]


def build() -> pd.DataFrame:
    df = load_resale()
    blocks = pd.read_parquet(OUT / "blocks.parquet").drop(columns=["footprint"])
    blocks = block_info(location_features(blocks))
    df = df.merge(blocks, on=["block", "street_name"], how="left")
    print(f"transactions: {len(df)}, without coordinates: {df.lat.isna().sum()}")

    idx = market_index(df)
    df = df.merge(idx, left_on="t", right_index=True, how="left")
    df = df[df.mkt_psm.notna()]

    # Comparable-sales features (relative to market, lagged).
    df = df.merge(lagged_group_psm(df, ["block", "street_name"], 24, "blk_rel"),
                  on=["block", "street_name", "t"], how="left")
    df = df.merge(lagged_group_psm(df, ["town", "flat_type"], 6, "town_type_rel"),
                  on=["town", "flat_type", "t"], how="left")
    return df


if __name__ == "__main__":
    df = build()
    df.to_parquet(OUT / "transactions.parquet")
    print(df.describe().T[["count", "mean", "min", "max"]].to_string())
