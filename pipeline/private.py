"""Private homes (URA transactions): build features, train, and export map data
for two segments: condo (apartments, condominiums, ECs) and landed (terrace,
semi-detached, detached, incl. strata landed).

Same approach as HDB: target is $/sqm relative to a lagged segment market
level, with lagged project and district comparables, forward-in-time backtest.
A project is identified by "project|street" (URA lumps many landed estates
under "LANDED HOUSING DEVELOPMENT", so street disambiguates).

For landed homes floor_area_sqm holds URA's area, which is land area for
typeOfArea=Land and strata floor area for strata landed.

Usage: private.py [condo|landed ...]   (default: both)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

import geocode
from features import lagged_group_psm, location_features
from score import Valuer
from train import train_and_save

ROOT = Path(__file__).resolve().parent.parent
DATA, WEB = ROOT / "data" / "processed", ROOT / "web" / "data"
ARCHIVE = ROOT / "data" / "raw" / "ura" / "archive.parquet"
TEST_MONTHS = 12
SALE_TYPES = {"1": "New Sale", "2": "Sub Sale", "3": "Resale"}
LANDED = {"Terrace", "Semi-detached", "Detached", "Strata Terrace", "Strata Semi-detached", "Strata Detached"}
CONDO = {"Apartment", "Condominium", "Executive Condominium"}
SIZE_BANDS = [(0, 50, "Under 50 sqm"), (50, 80, "50–80 sqm"), (80, 110, "80–110 sqm"),
              (110, 150, "110–150 sqm"), (150, 10_000, "150+ sqm")]

SPECS = {
    "condo": {"categorical": ["property_type", "market_segment", "district", "sale_type"],
              "numeric": ["floor_area_sqm", "floor", "freehold", "remaining_lease", "lease_start",
                          "lat", "lon", "mrt_dist", "school_dist", "schools_1km", "cbd_dist",
                          "proj_rel", "proj_rel_n", "dist_rel"],
              "increasing": ("floor",)},
    "landed": {"categorical": ["property_type", "market_segment", "district", "sale_type", "type_of_area"],
               "numeric": ["floor_area_sqm", "freehold", "remaining_lease", "lease_start",
                           "lat", "lon", "mrt_dist", "school_dist", "schools_1km", "cbd_dist",
                           "proj_rel", "proj_rel_n", "dist_rel"]},
}


def parse_floor(s: pd.Series) -> pd.Series:
    """'06-10' -> 8, 'B1-B5' -> -3, '-' (landed) -> NaN."""
    parts = s.str.replace("B", "-", regex=False).str.extract(r"^(-?\d+)-(-?\d+)$").astype(float)
    return parts.mean(axis=1)


def load() -> pd.DataFrame:
    df = pd.read_parquet(ARCHIVE)
    df = df[df.noOfUnits.astype(int) == 1].copy()  # drop bulk deals
    df["date"] = pd.to_datetime("20" + df.contractDate.str[2:] + "-" + df.contractDate.str[:2] + "-01")
    df["month"] = df.date.dt.strftime("%Y-%m")
    df["t"] = (df.date.dt.year - 2012) * 12 + df.date.dt.month - 1
    df["price"] = df.price.astype(float)
    df["floor_area_sqm"] = df.area.astype(float)
    df["psm"] = df.price / df.floor_area_sqm
    df["floor"] = parse_floor(df.floorRange)
    df["sale_type"] = df.typeOfSale.map(SALE_TYPES)
    df = df.rename(columns={"propertyType": "property_type", "marketSegment": "market_segment",
                            "typeOfArea": "type_of_area"})
    df["district"] = df.district.astype(int).astype(str).str.zfill(2)

    lease = df.tenure.str.extract(r"(\d+) y(?:ea)?rs lease commencing from (\d{4})").astype(float)
    years, start = lease[0], lease[1]
    df["freehold"] = ((df.tenure == "Freehold") | (years >= 900)).astype(int)
    df["lease_start"] = np.where(df.freehold == 1, np.nan, start)
    df["remaining_lease"] = np.where(df.freehold == 1, np.nan,
                                     years - (df.date.dt.year + (df.date.dt.month - 1) / 12 - start))
    df["pid"] = df.project + "|" + df.street
    df["segment"] = np.select([df.property_type.isin(CONDO), df.property_type.isin(LANDED)],
                              ["condo", "landed"], "other")
    return df


def project_locations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per project with lat/lon: URA's SVY21 x/y, else OneMap search."""
    proj = df.groupby("pid").agg(project=("project", "first"), street=("street", "first"),
                                 x=("x", "first"), y=("y", "first"), segment=("segment", "first")).reset_index()
    to_wgs = Transformer.from_crs(3414, 4326, always_xy=True)
    has = proj.x.notna() & proj.y.notna()
    lon, lat = to_wgs.transform(proj.loc[has, "x"].astype(float).to_numpy(), proj.loc[has, "y"].astype(float).to_numpy())
    proj.loc[has, "lon"], proj.loc[has, "lat"] = lon, lat
    proj["geo_source"] = np.where(has, "ura", None)

    missing = proj[~has]
    print(f"projects: {len(proj)}, without URA coordinates: {len(missing)} (querying OneMap)")
    for i, r in missing.iterrows():
        query = r.street if r.project == "LANDED HOUSING DEVELOPMENT" else r.project
        res = geocode.onemap(query) or geocode.onemap(r.street)
        if res:
            proj.loc[i, ["lat", "lon", "geo_source"]] = float(res[0]["LATITUDE"]), float(res[0]["LONGITUDE"]), "onemap"
    geocode.CACHE.write_text(json.dumps(geocode._cache))
    print("  geocode sources:", proj.geo_source.fillna("unresolved").value_counts().to_dict())
    return location_features(proj[proj.lat.notna()].reset_index(drop=True))


def fe_market_index(d: pd.DataFrame, min_months: int = 6) -> pd.Series:
    """Mix-adjusted market level per month t, using only sales before t.

    log psm = project effect + month effect + controls, fit within projects, so
    a month full of pricey new launches doesn't move the index. For each target
    month the model is refit on earlier data; the level is the mean of the last
    3 month effects, anchored to the median $/sqm of the first `min_months`."""
    d = d.sort_values("t")
    controls = pd.get_dummies(d[["sale_type", "property_type"]], drop_first=True, dtype=float)
    controls["log_area"] = np.log(d.floor_area_sqm)
    controls["floor"] = d.floor.fillna(0)
    y = np.log(d.psm).to_numpy()
    t0 = int(d.t.min())
    base = d[d.t < t0 + min_months].psm.median()
    out = {}
    for target in range(t0 + min_months, int(d.t.max()) + 2):
        m = (d.t < target).to_numpy()
        months = pd.get_dummies(d.t[m], dtype=float).iloc[:, 1:]  # first month = reference
        X = np.column_stack([months.to_numpy(), controls[m].to_numpy()])
        g = d.pid[m].to_numpy()
        Xd = X - pd.DataFrame(X).groupby(g).transform("mean").to_numpy()
        yd = y[m] - pd.Series(y[m]).groupby(g).transform("mean").to_numpy()
        beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
        effects = np.concatenate([[0.0], beta[:months.shape[1]]])
        out[target] = base * np.exp(effects[-3:]).mean()
    return pd.Series(out, name="mkt_psm")


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(fe_market_index(df), left_on="t", right_index=True, how="left")
    df = df[df.mkt_psm.notna()]
    df = df.merge(lagged_group_psm(df, ["pid"], 24, "proj_rel"), on=["pid", "t"], how="left")
    return df.merge(lagged_group_psm(df, ["district", "property_type"], 6, "dist_rel"),
                    on=["district", "property_type", "t"], how="left")


def build(seg: str, df: pd.DataFrame, locs: pd.DataFrame) -> pd.DataFrame:
    d = df[df.segment == seg]
    lo, hi = d.psm.quantile([0.001, 0.999])
    d = d[d.psm.between(lo, hi)]  # obvious data errors / whole-building records
    d = d.merge(locs.drop(columns=["project", "street", "x", "y", "segment"]), on="pid", how="inner")
    return add_market_features(d.sort_values("date").reset_index(drop=True))


def nowcast_frame(seg: str, d: pd.DataFrame) -> pd.DataFrame:
    """Typical units per project as of next month, valued as resales."""
    now_t = int(d.t.max()) + 1
    now = d.date.max() + pd.DateOffset(months=1)
    if seg == "condo":
        bands = pd.cut(d.floor_area_sqm, [b[0] for b in SIZE_BANDS] + [SIZE_BANDS[-1][1]],
                       labels=[b[2] for b in SIZE_BANDS], right=False)
        d = d.assign(unit=bands.astype(str))
    else:
        d = d.assign(unit=d.property_type)
    units = d.groupby(["pid", "unit"]).agg(
        floor_area_sqm=("floor_area_sqm", "median"), floor=("floor", "median"),
        property_type=("property_type", lambda s: s.mode().iat[0]), n_sales=("price", "size")).reset_index()
    if seg == "landed":
        units["type_of_area"] = d.groupby(["pid", "unit"]).type_of_area.agg(lambda s: s.mode().iat[0]).to_numpy()
    static = ["project", "street", "market_segment", "district", "tenure", "freehold", "lease_start",
              "lat", "lon", "mrt_dist", "mrt_station", "school_dist", "schools_1km", "cbd_dist"]
    last = d.sort_values("date").groupby("pid")[static + (["max_floor"] if "max_floor" in d else [])].last()
    if seg == "condo":
        last["max_floor"] = d.groupby("pid").floor.max()
    u = units.merge(last.reset_index(), on="pid")
    u = u[~u.project.str.contains("(DEMOLISHED)", regex=False)]
    u["t"], u["sale_type"] = now_t, "Resale"
    lease = u.tenure.str.extract(r"(\d+) y(?:ea)?rs lease")[0].astype(float)
    u["remaining_lease"] = np.where(u.freehold == 1, np.nan,
                                    lease - (now.year + (now.month - 1) / 12 - u.lease_start))
    u["mkt_psm"] = fe_market_index(d).loc[now_t]
    u = u.merge(lagged_group_psm(d, ["pid"], 24, "proj_rel"), on=["pid", "t"], how="left")
    u = u.merge(lagged_group_psm(d, ["district", "property_type"], 6, "dist_rel"),
                on=["district", "property_type", "t"], how="left")
    return u.assign(valuation_month=f"{now:%Y-%m}")


def export(seg: str, d: pd.DataFrame, u: pd.DataFrame, report: dict) -> None:
    tx = d[d.date >= d.date.max() - pd.DateOffset(years=5)].sort_values("date", ascending=False)
    tx_by = {k: g.head(25)[["month", "sale_type", "floorRange", "floor_area_sqm", "price", "property_type"]]
             .astype({"floor_area_sqm": float, "price": int}).values.tolist() for k, g in tx.groupby("pid")}
    band_order = {b[2]: i for i, b in enumerate(SIZE_BANDS)}
    out = []
    for pid, g in u.groupby("pid"):
        p = g.iloc[0]
        g = g.sort_values("unit", key=lambda s: s.map(band_order).fillna(0) if seg == "condo" else s)
        out.append({
            "id": pid, "project": p.project, "street": p.street, "district": p.district,
            "segment": p.market_segment, "tenure": p.tenure,
            "types": sorted(g.property_type.unique().tolist()),
            "lat": round(p.lat, 6), "lon": round(p.lon, 6),
            "floors": int(p.max_floor) if seg == "condo" and pd.notna(p.max_floor) else None,
            "mrt": p.mrt_station.title().replace(" Mrt Station", "").replace(" Lrt Station", " LRT"),
            "mrt_m": int(p.mrt_dist), "schools_1km": int(p.schools_1km),
            "psm": int(np.average(g.psm, weights=g.n_sales)),
            "units": [{"type": r.unit, "sqm": round(r.floor_area_sqm, 1),
                       "floor": None if pd.isna(r.floor) else int(r.floor),
                       "est": int(r.est), "lo": int(r.lo), "hi": int(r.hi), "n": int(r.n_sales)}
                      for r in g.itertuples()],
            "tx": tx_by.get(pid, []),
        })
    meta = {"valuation_month": u.valuation_month.iat[0], "latest_sale": f"{d.date.max():%Y-%m}",
            "eval": report["lightgbm"]}
    path = WEB / f"{seg}.json"
    path.write_text(json.dumps({"meta": meta, "projects": out}, separators=(",", ":")))
    print(f"[{seg}] wrote {len(out)} projects, {path.stat().st_size / 1e6:.1f} MB")


def main(segments: list[str]) -> None:
    df = load()
    locs = project_locations(df[df.segment.isin(segments)])
    for seg in segments:
        d = build(seg, df, locs)
        d.to_parquet(DATA / f"{seg}_transactions.parquet")
        spec = SPECS[seg]
        test_start = f"{d.date.max() - pd.DateOffset(months=TEST_MONTHS - 1):%Y-%m-01}"
        report = train_and_save(d.copy(), seg, spec["categorical"], spec["numeric"], test_start,
                                ["proj_rel", "dist_rel"], "property_type", spec.get("increasing", ()))
        frame = nowcast_frame(seg, d)
        frame.to_parquet(DATA / f"{seg}_nowcast.parquet")  # served by server.py for unit queries
        u = Valuer(seg).predict(frame)
        print(f"[{seg}] valuation month {u.valuation_month.iat[0]}: {len(u)} units in {u.pid.nunique()} projects")
        export(seg, d, u, report)


if __name__ == "__main__":
    main(sys.argv[1:] or ["condo", "landed"])
