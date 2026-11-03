"""Geocode every HDB block, plus primary schools and MRT stations.

Blocks: match each street name to a street code in the official HDB building
footprints (by overlap of block-number sets), then take the footprint centroid.
Blocks not in the footprints fall back to OneMap search (cached).

Outputs data/processed/{blocks,schools,mrt}.parquet
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
CACHE = RAW / "onemap_cache.json"
ONEMAP = "https://www.onemap.gov.sg/api/common/elastic/search"

_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def onemap(query: str) -> list[dict]:
    if query not in _cache:
        for attempt in range(5):
            r = requests.get(ONEMAP, params={"searchVal": query, "returnGeom": "Y",
                                             "getAddrDetails": "Y", "pageNum": 1}, timeout=30)
            if r.status_code == 200:
                _cache[query] = r.json().get("results", [])
                break
            time.sleep(2 ** attempt)
        else:
            r.raise_for_status()
        time.sleep(0.25)  # OneMap allows ~250 req/min
        if len(_cache) % 50 == 0:
            CACHE.write_text(json.dumps(_cache))
    return _cache[query]


def ring_centroid(ring: list) -> tuple[float, float, float]:
    """Shoelace centroid of a lon/lat ring; returns (lon, lat, signed area)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        c = x0 * y1 - x1 * y0
        a += c
        cx += (x0 + x1) * c
        cy += (y0 + y1) * c
    a /= 2
    return cx / (6 * a), cy / (6 * a), a


def footprint_index() -> dict:
    """(street_code, block) -> {lon, lat, postal, polygons}"""
    g = json.loads((RAW / "hdb_buildings.geojson").read_text())
    acc = defaultdict(lambda: {"sx": 0.0, "sy": 0.0, "a": 0.0, "polys": [], "postal": None})
    for f in g["features"]:
        p, geom = f["properties"], f["geometry"]
        key = (p["ST_COD"], str(p["BLK_NO"]).strip().upper())
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        e = acc[key]
        e["postal"] = p.get("POSTAL_COD")
        for poly in polys:
            ring = [pt[:2] for pt in poly[0]]
            x, y, a = ring_centroid(ring)
            e["sx"] += x * abs(a)
            e["sy"] += y * abs(a)
            e["a"] += abs(a)
            e["polys"].append(ring)
    return {k: {"lon": v["sx"] / v["a"], "lat": v["sy"] / v["a"], "postal": v["postal"],
                "polys": v["polys"]} for k, v in acc.items() if v["a"] > 0}


def match_streets(streets: dict[str, set], codes: dict[str, set]) -> dict[str, str]:
    """street name -> street code with the highest Jaccard overlap of block numbers."""
    out = {}
    for s, blocks in streets.items():
        best, score = None, 0.0
        for c, cblocks in codes.items():
            j = len(blocks & cblocks) / len(blocks | cblocks)
            if j > score:
                best, score = c, j
        if best:
            out[s] = best
    return out


def geocode_blocks() -> pd.DataFrame:
    pi = pd.read_csv(RAW / "hdb_property_info.csv", dtype=str)
    resale = pd.concat(pd.read_csv(p, usecols=["block", "street_name", "town"], dtype=str)
                       for p in RAW.glob("resale_*.csv"))
    keys = pd.concat([
        pi.rename(columns={"blk_no": "block", "street": "street_name"})[["block", "street_name"]],
        resale[["block", "street_name"]],
    ])
    keys["block"] = keys.block.str.strip().str.upper()
    keys["street_name"] = keys.street_name.str.strip().str.upper()
    keys = keys.drop_duplicates().reset_index(drop=True)

    fp = footprint_index()
    codes = defaultdict(set)
    for c, b in fp:
        codes[c].add(b)
    streets = keys.groupby("street_name").block.apply(set).to_dict()
    street_code = match_streets(streets, codes)

    rows, misses = [], []
    for block, street in keys.itertuples(index=False):
        hit = fp.get((street_code.get(street), block))
        if hit:
            rows.append({"block": block, "street_name": street, "lat": hit["lat"], "lon": hit["lon"],
                         "postal": hit["postal"], "source": "footprint",
                         "footprint": json.dumps(hit["polys"])})
        else:
            misses.append((block, street))

    print(f"footprint matched {len(rows)}/{len(keys)}; querying OneMap for {len(misses)}")
    for block, street in misses:
        res = [r for r in onemap(f"{block} {street}") if r["BLK_NO"].upper() == block]
        if res:
            r = res[0]
            rows.append({"block": block, "street_name": street, "lat": float(r["LATITUDE"]),
                         "lon": float(r["LONGITUDE"]), "postal": r["POSTAL"], "source": "onemap",
                         "footprint": None})
        else:
            print(f"  unresolved: {block} {street}")
    CACHE.write_text(json.dumps(_cache))
    return pd.DataFrame(rows)


def validate(blocks: pd.DataFrame, n: int = 60) -> None:
    """Spot-check footprint coordinates against OneMap; report distance errors."""
    sample = blocks[blocks.source == "footprint"].sample(n, random_state=0)
    errs = []
    for r in sample.itertuples():
        res = [x for x in onemap(f"{r.block} {r.street_name}") if x["BLK_NO"].upper() == r.block]
        if res:
            dlat = (float(res[0]["LATITUDE"]) - r.lat) * 111_000
            dlon = (float(res[0]["LONGITUDE"]) - r.lon) * 111_000
            errs.append((dlat ** 2 + dlon ** 2) ** 0.5)
    CACHE.write_text(json.dumps(_cache))
    s = pd.Series(errs)
    print(f"validation vs OneMap: n={len(s)} median={s.median():.0f}m p95={s.quantile(.95):.0f}m "
          f"max={s.max():.0f}m >200m={int((s > 200).sum())}")


def geocode_schools() -> pd.DataFrame:
    s = pd.read_csv(RAW / "schools.csv", dtype=str)
    s = s[s.mainlevel_code.isin(["PRIMARY", "MIXED LEVEL (P1-S4)"])]
    rows = []
    for r in s.itertuples():
        res = onemap(r.postal_code.zfill(6))
        if res:
            rows.append({"school": r.school_name, "lat": float(res[0]["LATITUDE"]),
                         "lon": float(res[0]["LONGITUDE"])})
        else:
            print(f"  unresolved school: {r.school_name}")
    CACHE.write_text(json.dumps(_cache))
    return pd.DataFrame(rows)


def mrt_stations() -> pd.DataFrame:
    g = json.loads((RAW / "mrt_exits.geojson").read_text())
    exits = pd.DataFrame([{"station": f["properties"]["STATION_NA"],
                           "lon": f["geometry"]["coordinates"][0],
                           "lat": f["geometry"]["coordinates"][1]} for f in g["features"]])
    return exits


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = geocode_blocks()
    blocks.to_parquet(OUT / "blocks.parquet")
    print(blocks.source.value_counts().to_dict())
    validate(blocks)
    schools = geocode_schools()
    schools.to_parquet(OUT / "schools.parquet")
    print(f"schools: {len(schools)}")
    mrt = mrt_stations()
    mrt.to_parquet(OUT / "mrt_exits.parquet")
    print(f"mrt exits: {len(mrt)}, stations: {mrt.station.nunique()}")
