"""Generate Lorong's demo data from the sg-homes models and real transaction data.

Listings are synthetic: each one sits on a real HDB block or URA project, the
unit (type, size, floor) is drawn from what actually sells there, and its value
comes from sg-homes' unit valuation functions (the same code path as its API).
The asking price is that estimate times a random seller markup, so some
listings are above and some below estimate. Agents and agencies are fictional.

Outputs (data/): listings.json, agents.json, projects.json, trends.json
Run with the sg-homes venv: $SG_HOMES_DIR/.venv/bin/python generate.py
"""
import json
import math
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# The sg-homes checkout with models and data: SG_HOMES_DIR, else the repo root, else a sibling folder.
SG = next((Path(c).resolve() for c in [os.environ.get("SG_HOMES_DIR"), ROOT.parent, ROOT.parent.parent / "sg-homes"]
           if c and (Path(c) / "models" / "hdb_serving.json").exists()), None)
if SG is None:
    raise SystemExit("sg-homes models not found; set SG_HOMES_DIR to the sg-homes checkout")
sys.path.insert(0, str(SG / "pipeline"))
import server as sg  # noqa: E402  (sg-homes valuation API functions; loads models on import)

OUT = ROOT / "data"
SQFT = 10.7639
SEED = 7
COUNTS = {"hdb": 1400, "condo": 1200, "landed": 300}
RENT_SHARE = 0.25  # share of HDB/condo listings offered for rent instead
TODAY = date.today()

HDB_BEDS = {"1 ROOM": (1, 1), "2 ROOM": (1, 1), "3 ROOM": (2, 2), "4 ROOM": (3, 2), "5 ROOM": (3, 2),
            "EXECUTIVE": (3, 3), "MULTI-GENERATION": (4, 3)}
ROOMS = ["Living room", "Kitchen", "Master bedroom", "Bedroom", "Bathroom", "View", "Exterior", "Dining"]
CONDO_FACILITIES = ["Swimming pool", "Gym", "BBQ pits", "Function room", "Playground", "Tennis court",
                    "Clubhouse", "Jacuzzi", "Sky garden", "Covered parking", "24-hour security", "Lap pool"]
FEATURES = ["Renovated", "Corner unit", "Unblocked view", "Near MRT", "Move-in condition", "Bright and airy",
            "Pet friendly", "High ceiling", "Balcony", "Walk-in wardrobe", "Utility room", "Squarish layout"]

# Fictional agents: names are built from common Singapore name parts, not real people.
GIVEN = ["Wei Ling", "Jun Hao", "Siti", "Arjun", "Mei Xin", "Hafiz", "Priya", "Darren", "Nur Aisyah",
         "Kelvin", "Rachel", "Vikram", "Jasmine", "Faizal", "Yvonne", "Marcus", "Shu Fen", "Imran",
         "Clarissa", "Ravi", "Hui Min", "Zul", "Denise", "Benjamin"]
FAMILY = ["Tan", "Lim", "Ng", "Rahman", "Pillai", "Ong", "Goh", "Ismail", "Nair", "Chua", "Koh",
          "Yusof", "Teo", "Menon", "Lee", "Wong", "Abdullah", "Sim"]
AGENCIES = ["Harbourline Realty", "Kopi Keys Property", "Banyan Home Partners", "Meridian Estates SG",
            "Little Red Dot Realty", "Tembusu Property Group"]


def title(s: str) -> str:
    return " ".join(w.capitalize() if not w.isdigit() else w for w in s.lower().split())


def make_agents(rng: random.Random, n: int = 48) -> list[dict]:
    names, agents = set(), []
    while len(agents) < n:
        name = f"{rng.choice(GIVEN)} {rng.choice(FAMILY)}"
        if name in names:
            continue
        names.add(name)
        agents.append({"id": f"A{len(agents) + 1:03d}", "name": name, "agency": rng.choice(AGENCIES),
                       "years": rng.randint(1, 22), "rating": round(rng.uniform(4.1, 5.0), 1),
                       "reviews": rng.randint(3, 240),
                       "specialty": rng.choice(["HDB resale", "Condos", "Landed", "New launches", "Rentals"])})
    return agents


def floor_label(floor: float, top: float | None) -> str:
    if not top or top <= 3:
        return "Low floor"
    r = floor / top
    return "High floor" if r > 0.66 else "Mid floor" if r > 0.33 else "Low floor"


def ask_price(est: float, markup_mu: float, markup_sd: float, step: int, rng: np.random.Generator) -> int:
    return int(round(est * math.exp(rng.normal(markup_mu, markup_sd)) / step) * step)


def describe(l: dict, rng: random.Random) -> str:
    where = l["town"] if l["kind"] == "hdb" else f"District {l['district']}"
    if l["kind"] == "hdb":
        first = f"{l['beds']}-bedroom HDB flat at {l['address']}, {l['floor_label'].lower()}."
    elif l["kind"] == "condo":
        first = f"{l['beds']}-bedroom unit at {l['name']}, {l['floor_label'].lower()}, {l['sqft']:,} sqft."
    else:
        place = title(l["street"]) if l["name"].startswith("Landed") else l["name"]
        first = f"{l['property_type']} in {place}, land area {l['sqft']:,} sqft."
    mrt = f"About {l['mrt_m']} m to {l['mrt']} MRT." if l["mrt_m"] < 800 else f"{l['mrt']} MRT is the nearest station."
    extras = rng.sample(l["features"], k=min(2, len(l["features"])))
    tail = rng.choice([
        "Viewing by appointment.", "Motivated seller.", "Well kept by owner-occupiers.",
        "Flexible on handover timing.", "Photos are placeholders in this demo.",
    ])
    lease = "" if l["tenure"] == "Freehold" else f" {l['remaining_lease']} years left on the lease."
    return " ".join([first, f"{', '.join(extras)}." if extras else "", mrt,
                     f"{l['schools_1km']} primary school(s) within 1 km of {where}.", lease.strip(), tail]).replace("  ", " ")


def images(rng: random.Random, kind: str) -> list[dict]:
    rooms = ["Exterior" if kind == "landed" else "Living room"] + rng.sample(ROOMS[1:], k=rng.randint(4, 6))
    return [{"room": r, "seed": rng.randint(0, 10_000)} for r in rooms]


def remaining(lease_start: float | None, years: int = 99) -> int | None:
    if lease_start is None or (isinstance(lease_start, float) and math.isnan(lease_start)):
        return None
    return int(years - (TODAY.year - lease_start))


def gen_hdb(n: int, rng: random.Random, nrng: np.random.Generator) -> list[dict]:
    blocks = json.loads((SG / "web/data/blocks.json").read_text())["blocks"]
    weights = [sum(u["n"] for u in b["units"]) for b in blocks]
    out = []
    while len(out) < n:
        b = rng.choices(blocks, weights)[0]
        u = rng.choices(b["units"], [x["n"] for x in b["units"]])[0]
        top = b["floors"] or 12
        storey = rng.randint(1, top)
        sqm = round(u["sqm"] + rng.choice([0, 0, 0, -2, 2, 3]), 0)
        status, val = sg.estimate_hdb({"id": b["id"], "type": u["type"], "storey": str(storey), "sqm": str(sqm)})
        if status != 200 or val["warnings"]:
            continue
        beds, baths = HDB_BEDS.get(u["type"], (3, 2))
        out.append({
            "kind": "hdb", "property_type": "HDB " + title(u["type"]), "name": f"Blk {b['block']} {title(b['street'])}",
            "address": f"Blk {b['block']} {title(b['street'])}", "block": b["block"], "street": b["street"],
            "town": title(b["town"]), "district": None, "segment": None,
            "lat": b["lat"], "lon": b["lon"], "sqm": sqm, "beds": beds, "baths": baths,
            "floor": storey, "floor_label": floor_label(storey, top), "floors": top,
            "tenure": "99-year leasehold", "lease_start": b["lease_start"],
            "remaining_lease": remaining(b["lease_start"]), "mrt": b["mrt"], "mrt_m": b["mrt_m"],
            "schools_1km": b["schools_1km"], "ref": {"kind": "hdb", "id": b["id"], "unit": u["type"]},
            "valuation": val, "_markup": (0.03, 0.05, 1000),
            "features": rng.sample(FEATURES, k=rng.randint(2, 4)),
        })
    return out


def gen_private(seg: str, n: int, rng: random.Random) -> list[dict]:
    projects = json.loads((SG / f"web/data/{seg}.json").read_text())["projects"]
    projects = [p for p in projects if any(t[0] >= f"{TODAY.year - 3}" for t in p["tx"])]  # still trading
    weights = [sum(u["n"] for u in p["units"]) for p in projects]
    out = []
    while len(out) < n:
        p = rng.choices(projects, weights)[0]
        u = rng.choices(p["units"], [x["n"] for x in p["units"]])[0]
        if seg == "condo":
            top = p["floors"] or 10
            floor = rng.randint(1, top)
            sqm = round(u["sqm"] * rng.uniform(0.9, 1.1), 0)
            status, val = sg.estimate_private({"seg": "condo", "id": p["id"], "sqm": str(sqm), "floor": str(floor)})
            beds = 1 if sqm < 50 else 2 if sqm < 80 else 3 if sqm < 115 else 4 if sqm < 160 else 5
            baths = max(1, beds - (1 if beds >= 3 else 0))
            ptype, markup = p["types"][0], (0.04, 0.06, 10_000)
        else:
            top, floor = None, None
            sqm = round(u["sqm"] * rng.uniform(0.85, 1.2), 0)
            status, val = sg.estimate_private({"seg": "landed", "id": p["id"], "type": u["type"], "sqm": str(sqm)})
            beds = rng.randint(4, 6) if "Detached" in u["type"] else rng.randint(4, 5)
            baths = beds + rng.randint(0, 1)
            ptype, markup = u["type"], (0.05, 0.09, 50_000)
        if status != 200 or val["warnings"]:
            continue
        lease_years = 99
        if p["tenure"] != "Freehold":
            try:
                lease_years = int(p["tenure"].split()[0])
            except ValueError:
                pass
        lease_start = None
        if "commencing from" in p["tenure"]:
            lease_start = int(p["tenure"].rsplit(" ", 1)[1])
        freehold = p["tenure"] == "Freehold" or lease_years >= 900
        name = f"Landed homes, {title(p['street'])}" if p["project"] == "LANDED HOUSING DEVELOPMENT" else title(p["project"])
        out.append({
            "kind": seg, "property_type": ptype, "name": name,
            "address": f"{name}, {title(p['street'])}" if not name.startswith("Landed") else title(p["street"]),
            "block": None, "street": p["street"], "town": None, "district": p["district"], "segment": p["segment"],
            "lat": p["lat"], "lon": p["lon"], "sqm": sqm, "beds": beds, "baths": baths,
            "floor": floor, "floor_label": floor_label(floor, top) if floor else None, "floors": top,
            "tenure": "Freehold" if freehold else f"{lease_years}-year leasehold",
            "lease_start": lease_start, "remaining_lease": None if freehold else remaining(lease_start, lease_years),
            "mrt": p["mrt"], "mrt_m": p["mrt_m"], "schools_1km": p["schools_1km"],
            "ref": {"kind": seg, "id": p["id"], "unit": u["type"]}, "valuation": val, "_markup": markup,
            "features": rng.sample(FEATURES, k=rng.randint(2, 4)),
            "facilities": rng.sample(CONDO_FACILITIES, k=rng.randint(5, 9)) if seg == "condo" else [],
        })
    return out


def finalize(listings: list[dict], agents: list[dict], rng: random.Random, nrng: np.random.Generator) -> None:
    for i, l in enumerate(listings):
        mu, sd, step = l.pop("_markup")
        l["id"] = f"L{100001 + i}"
        l["sqft"] = int(round(l["sqm"] * SQFT))
        est = l["valuation"]["est"]
        rentable = l["kind"] != "landed" and rng.random() < RENT_SHARE
        if rentable:
            # Synthetic asking rent from a gross yield; there's no rent model behind it.
            yld = nrng.normal(0.045 if l["kind"] == "hdb" else 0.034, 0.004)
            l["mode"], l["price"] = "rent", int(round(est * yld / 12 / 50) * 50)
        else:
            l["mode"], l["price"] = "sale", ask_price(est, mu, sd, step, nrng)
        l["psf"] = round(l["price"] / l["sqft"], 2 if rentable else 0)
        l["furnishing"] = rng.choice(["Unfurnished", "Partially furnished", "Fully furnished"])
        l["facing"] = rng.choice(["North", "South", "East", "West", "North-east", "South-west"])
        l["listed"] = (TODAY - timedelta(days=int(nrng.exponential(18)))).isoformat()
        l["agent"] = rng.choice(agents)["id"]
        l["images"] = images(rng, l["kind"])
        l["description"] = describe(l, rng)
        l["views"] = int(nrng.lognormal(5, 0.8))


def trends() -> dict:
    out = {}
    hdb = pd.read_parquet(SG / "data/processed/transactions.parquet", columns=["month", "psm", "flat_type", "town"])
    hdb = hdb[hdb.month >= "2016-01"]
    out["hdb"] = monthly(hdb)
    out["hdb_by_type"] = {t: monthly(g) for t, g in hdb.groupby("flat_type") if len(g) > 2000}
    for seg in ["condo", "landed"]:
        d = pd.read_parquet(SG / f"data/processed/{seg}_transactions.parquet",
                            columns=["month", "psm", "market_segment", "sale_type"])
        out[seg] = monthly(d)
        out[f"{seg}_by_region"] = {r: monthly(g) for r, g in d.groupby("market_segment")}
        if seg == "condo":
            out["condo_resale"] = monthly(d[d.sale_type == "Resale"])
            out["condo_new"] = monthly(d[d.sale_type == "New Sale"])
    return out


def monthly(d: pd.DataFrame) -> dict:
    g = d.groupby("month").psm.agg(["median", "size"])
    return {"month": g.index.tolist(), "psf": (g["median"] / SQFT).round(0).astype(int).tolist(),
            "volume": g["size"].astype(int).tolist()}


def projects_index() -> list[dict]:
    """Condo directory: every project still trading, with recent price ranges and a
    new-launch flag (new sales in the last 12 months)."""
    projects = json.loads((SG / "web/data/condo.json").read_text())["projects"]
    cutoff = f"{TODAY.year - 1}-{TODAY.month:02d}"
    out = []
    for p in projects:
        recent = [t for t in p["tx"] if t[0] >= f"{TODAY.year - 3}"]
        if not recent:
            continue
        psf = [t[4] / t[3] / SQFT for t in recent]
        new = [t for t in recent if t[1] == "New Sale" and t[0] >= cutoff]
        out.append({"id": p["id"], "name": title(p["project"]), "street": title(p["street"]),
                    "district": p["district"], "segment": p["segment"], "tenure": p["tenure"],
                    "types": p["types"], "lat": p["lat"], "lon": p["lon"], "floors": p["floors"],
                    "mrt": p["mrt"], "mrt_m": p["mrt_m"], "psf_lo": int(min(psf)), "psf_hi": int(max(psf)),
                    "psf_est": int(p["psm"] / SQFT), "sales_3y": len(recent), "new_launch": bool(new),
                    "new_sales_12m": len(new), "last_sale": recent[0][0]})
    return out


def hero_points() -> list[list]:
    """Every valued block/project as [lon, lat, height_m, psf, kind] for the 3D home-page city."""
    out = []
    for b in json.loads((SG / "web/data/blocks.json").read_text())["blocks"]:
        out.append([round(b["lon"], 5), round(b["lat"], 5), (b["floors"] or 4) * 3, int(b["psm"] / SQFT), 0])
    for k, seg in [(1, "condo"), (2, "landed")]:
        for p in json.loads((SG / f"web/data/{seg}.json").read_text())["projects"]:
            h = (p["floors"] or 4) * 3.2 if seg == "condo" else 9
            out.append([round(p["lon"], 5), round(p["lat"], 5), round(h), int(p["psm"] / SQFT), k])
    return out


def areas() -> list[dict]:
    """Neighbourhood summaries: HDB towns (from block estimates) with listing counts filled in later."""
    blocks = json.loads((SG / "web/data/blocks.json").read_text())["blocks"]
    df = pd.DataFrame([{"town": title(b["town"]), "psf": b["psm"] / SQFT, "lat": b["lat"], "lon": b["lon"],
                        "lease_start": b["lease_start"]} for b in blocks])
    g = df.groupby("town").agg(psf=("psf", "median"), blocks=("psf", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
                               lease_start=("lease_start", "median")).reset_index()
    return [{"town": r.town, "psf": int(r.psf), "blocks": int(r.blocks), "lat": round(r.lat, 5), "lon": round(r.lon, 5),
             "median_lease_start": int(r.lease_start)} for r in g.itertuples()]


def eval_details() -> dict:
    """Backtest results from the sg-homes eval reports, for the methodology page."""
    out = {}
    for k in ["hdb", "condo", "landed"]:
        r = json.loads((SG / f"models/{k}_eval_report.json").read_text())
        out[k] = {"model": r["lightgbm"], "baseline": r["baseline_comps"],
                  "coverage": r["interval_p10_p90_calibrated"]["coverage"],
                  "test_months": sorted(r["mape_by_test_month"])[::len(r["mape_by_test_month"]) - 1],
                  "top_features": list(r["feature_importance_gain_pct"].items())[:5]}
    fc = json.loads((SG / "models/forecast.json").read_text())["backtest"]
    out["lease_decay"] = {k: fc[k] for k in ["pairs", "horizon_years", "rmse_no_change", "rmse_decay_model", "corr_pred_actual"]}
    return out


def main() -> None:
    rng, nrng = random.Random(SEED), np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)
    agents = make_agents(rng)
    listings = gen_hdb(COUNTS["hdb"], rng, nrng)
    print(f"hdb: {len(listings)}")
    for seg in ["condo", "landed"]:
        listings += gen_private(seg, COUNTS[seg], rng)
        print(f"{seg}: {sum(l['kind'] == seg for l in listings)}")
    rng.shuffle(listings)
    finalize(listings, agents, rng, nrng)
    meta = {"generated": TODAY.isoformat(),
            "valuation_month": json.loads((SG / "web/data/condo.json").read_text())["meta"]["valuation_month"],
            "eval": {k: json.loads((SG / f"models/{k}_eval_report.json").read_text())["lightgbm"]["MdAPE"]
                     for k in ["hdb", "condo", "landed"]},
            "eval_details": eval_details(),
            "synthetic": True}
    (OUT / "hero.json").write_text(json.dumps(hero_points(), separators=(",", ":")))
    (OUT / "areas.json").write_text(json.dumps(areas(), separators=(",", ":")))
    (OUT / "listings.json").write_text(json.dumps({"meta": meta, "listings": listings}, separators=(",", ":")))
    (OUT / "agents.json").write_text(json.dumps(agents, indent=1))
    (OUT / "projects.json").write_text(json.dumps(projects_index(), separators=(",", ":")))
    (OUT / "trends.json").write_text(json.dumps(trends(), separators=(",", ":")))
    sale = [l for l in listings if l["mode"] == "sale"]
    gap = np.array([l["price"] / l["valuation"]["est"] - 1 for l in sale])
    print(f"listings: {len(listings)} ({len(sale)} sale, {len(listings) - len(sale)} rent); "
          f"asking vs estimate median {np.median(gap):+.1%}, below estimate {np.mean(gap < 0):.0%}")
    for f in sorted(OUT.glob("*.json")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
