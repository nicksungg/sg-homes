"""Download URA private residential transactions (all 4 batches) into a dated
snapshot under data/raw/ura/<YYYY-MM-DD>/, and merge every snapshot into
data/raw/ura/archive.parquet so history accumulates beyond URA's 5-year window.

Needs URA_ACCESS_KEY in sg-homes/.env (register at
https://eservice.ura.gov.sg/maps/api/reg.html).
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
URA = ROOT / "data" / "raw" / "ura"
BASE = "https://eservice.ura.gov.sg/uraDataService"
HEADERS = {"User-Agent": "Mozilla/5.0"}  # URA rejects the default python-requests agent


def access_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("URA_ACCESS_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("URA_ACCESS_KEY missing from .env")


def token(key: str) -> str:
    r = requests.get(f"{BASE}/insertNewToken/v1", headers={**HEADERS, "AccessKey": key}, timeout=60).json()
    if r.get("Status") != "Success":
        raise SystemExit(f"URA token request failed: {r.get('Message')}")
    return r["Result"]


def fetch_batches(snapshot: Path) -> None:
    key = access_key()
    tok = token(key)
    snapshot.mkdir(parents=True, exist_ok=True)
    for batch in range(1, 5):
        out = snapshot / f"transactions_batch{batch}.json"
        if out.exists():
            continue
        r = requests.get(f"{BASE}/invokeUraDS/v1", params={"service": "PMI_Resi_Transaction", "batch": batch},
                         headers={**HEADERS, "AccessKey": key, "Token": tok}, timeout=300)
        r.raise_for_status()
        body = r.json()
        if body.get("Status") != "Success":
            raise SystemExit(f"batch {batch} failed: {body.get('Message')}")
        out.write_text(json.dumps(body["Result"]))
        print(f"batch {batch}: {len(body['Result'])} projects")


def flatten(snapshot: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(snapshot.glob("transactions_batch*.json")):
        for p in json.loads(f.read_text()):
            proj = {k: v for k, v in p.items() if k != "transaction"}
            rows += [{**proj, **t} for t in p.get("transaction", [])]
    return pd.DataFrame(rows).assign(snapshot=snapshot.name)


def merge_archive() -> pd.DataFrame:
    """Union of all snapshots. A transaction is identified by all its fields;
    URA may revise records, so the latest snapshot wins for exact duplicates."""
    df = pd.concat(flatten(s) for s in sorted(URA.glob("20*")) if s.is_dir())
    key = [c for c in df.columns if c != "snapshot"]
    df = df.astype({c: str for c in key}).sort_values("snapshot").drop_duplicates(key, keep="last")
    df.to_parquet(URA / "archive.parquet")
    return df


if __name__ == "__main__":
    fetch_batches(URA / dt.date.today().isoformat())
    df = merge_archive()
    print(f"archive: {len(df)} transactions from {df.snapshot.nunique()} snapshot(s)")
