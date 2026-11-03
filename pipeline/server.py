"""Serve the map (web/) and unit valuation APIs.

GET /api/estimate?id=<block>|<street>&type=<flat type>&storey=<n>&sqm=<area>          (HDB)
GET /api/estimate_private?seg=condo&id=<project>|<street>&sqm=<area>&floor=<n>
GET /api/estimate_private?seg=landed&id=<project>|<street>&type=<property type>&sqm=<land area>
-> {"est", "lo", "hi", "psm", "typical_sqm", "warnings"}

Uses the same nowcast features and models as the batch scoring, overriding only
the unit's own attributes.
"""
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from private import SIZE_BANDS
from score import DATA, ROOT, Valuer

HDB = pd.read_parquet(DATA / "nowcast_units.parquet").set_index(["block", "street_name", "flat_type"])
PRIVATE = {seg: pd.read_parquet(DATA / f"{seg}_nowcast.parquet") for seg in ["condo", "landed"]}
SIZES_SOLD = pd.read_parquet(DATA / "condo_transactions.parquet", columns=["pid", "floor_area_sqm"]) \
    .groupby("pid").floor_area_sqm.agg(["min", "max"])
VALUERS = {seg: Valuer(seg) for seg in ["hdb", "condo", "landed"]}
SQM_RANGE = {"hdb": (25, 300), "condo": (20, 1000), "landed": (50, 5000)}


def result(seg: str, row: pd.DataFrame, typical: float, warnings: list[str]) -> tuple[int, dict]:
    r = VALUERS[seg].predict(row).iloc[0]
    return 200, {"est": int(r.est), "lo": int(r.lo), "hi": int(r.hi), "psm": int(r.psm),
                 "typical_sqm": typical, "warnings": warnings}


def size_warning(sqm: float, typical: float, what: str) -> list[str]:
    if abs(sqm - typical) / typical > 0.15:
        return [f"Size differs a lot from this project's typical {what} ({typical:.0f} sqm); "
                "estimate is less reliable."]
    return []


def estimate_hdb(params: dict) -> tuple[int, dict]:
    try:
        block, street = params["id"].split("|", 1)
        flat_type, storey, sqm = params["type"], float(params["storey"]), float(params["sqm"])
    except (KeyError, ValueError):
        return 400, {"error": "need id=<block>|<street>, type, storey, sqm"}
    if (block, street, flat_type) not in HDB.index:
        return 404, {"error": f"no {flat_type} resales in this block to value from"}
    row = HDB.loc[[(block, street, flat_type)]].reset_index()
    max_floor = row.max_floor_lvl.iat[0]
    lo, hi = SQM_RANGE["hdb"]
    if not lo <= sqm <= hi or storey < 1 or (pd.notna(max_floor) and storey > max_floor):
        return 400, {"error": f"storey must be 1–{max_floor:.0f}, sqm {lo}–{hi}"}
    typical = float(row.floor_area_sqm.iat[0])
    warnings = size_warning(sqm, typical, flat_type.lower())
    row["storey"], row["floor_area_sqm"] = storey, sqm
    return result("hdb", row, typical, warnings)


def estimate_private(params: dict) -> tuple[int, dict]:
    seg = params.get("seg")
    if seg not in PRIVATE:
        return 400, {"error": "seg must be condo or landed"}
    try:
        pid, sqm = params["id"], float(params["sqm"])
    except (KeyError, ValueError):
        return 400, {"error": "need id=<project>|<street> and sqm"}
    rows = PRIVATE[seg][PRIVATE[seg].pid == pid]
    if rows.empty:
        return 404, {"error": "no sales in this project to value from"}
    lo, hi = SQM_RANGE[seg]
    if not lo <= sqm <= hi:
        return 400, {"error": f"sqm must be {lo}–{hi}"}

    if seg == "condo":
        try:
            floor = float(params["floor"])
        except (KeyError, ValueError):
            return 400, {"error": "need floor"}
        top = rows.max_floor.iat[0]
        if floor < -5 or (pd.notna(top) and floor > top + 5):
            return 400, {"error": f"floor must be up to about {top:.0f} (highest sold here)"}
        # Use the project's size band containing sqm, else the nearest band sold here.
        order = {b[2]: (b[0] + min(b[1], 300)) / 2 for b in SIZE_BANDS}
        row = rows.iloc[[(rows.unit.map(order) - sqm).abs().argmin()]].copy()
        lo_seen, hi_seen = SIZES_SOLD.loc[pid]
        warnings = [] if 0.9 * lo_seen <= sqm <= 1.1 * hi_seen else [
            f"Size is outside what has sold here ({lo_seen:.0f}–{hi_seen:.0f} sqm); estimate is less reliable."]
        typical = float(row.floor_area_sqm.iat[0])
        row["floor_area_sqm"], row["floor"] = sqm, floor
        return result(seg, row, typical, warnings)

    ptype = params.get("type")
    row = rows[rows.unit == ptype].copy()
    if row.empty:
        return 404, {"error": f"no {ptype} sales in this street/estate to value from"}
    typical = float(row.floor_area_sqm.iat[0])
    warnings = size_warning(sqm, typical, ptype.lower())
    row["floor_area_sqm"] = sqm
    return result(seg, row, typical, warnings)


ROUTES = {"/api/estimate": estimate_hdb, "/api/estimate_private": estimate_private}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path not in ROUTES:
            return super().do_GET()
        status, body = ROUTES[url.path]({k: v[0] for k, v in parse_qs(url.query).items()})
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"serving http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "web"))).serve_forever()
