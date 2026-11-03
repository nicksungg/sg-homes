"""Check Lorong end to end (page, data, a real valuation); restart it locally if broken,
and record whether the public URL answers. Run every two minutes by lorong-health.timer."""
import datetime as dt
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / "data/logs/health.json"
PUBLIC = json.loads((ROOT / "deploy/public-host.json").read_text()).get("public_url")
VALUE = {"kind": "hdb", "id": "406|ANG MO KIO AVE 10", "type": "2 ROOM", "storey": "11", "sqm": "44"}


def check(base: str) -> int:
    with urllib.request.urlopen(base + "/", timeout=15) as r:
        assert b"Lorong" in r.read(), "home page missing"
    with urllib.request.urlopen(base + "/data/listings.json", timeout=30) as r:
        assert len(json.load(r)["listings"]) > 1000, "listings missing"
    req = urllib.request.Request(base + "/api/value", data=json.dumps(VALUE).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        v = json.load(r)
    assert 0 < v["lo"] <= v["est"] <= v["hi"], "invalid valuation"
    return v["est"]


state = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
try:
    state["local_estimate"], state["local"] = check("http://127.0.0.1:8780"), "healthy"
except Exception as err:  # noqa: BLE001
    state["local_error"] = repr(err)
    subprocess.run(["systemctl", "--user", "restart", "lorong.service"], check=True)
    time.sleep(20)  # models and map data take a few seconds to load
    try:
        state["local_estimate"], state["local"] = check("http://127.0.0.1:8780"), "recovered"
    except Exception as retry:  # noqa: BLE001
        state["local"], state["recovery_error"] = "failed", repr(retry)
if PUBLIC:
    try:
        state["public_estimate"], state["public"] = check(PUBLIC.rstrip("/")), "healthy"
    except Exception as err:  # noqa: BLE001
        state["public"], state["public_error"] = "unreachable", repr(err)
else:
    state["public"] = "not-configured"
STATUS.parent.mkdir(parents=True, exist_ok=True)
STATUS.write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(state), flush=True)
if state["local"] == "failed" or state["public"] == "unreachable":
    raise SystemExit(1)
