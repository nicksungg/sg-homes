"""Package the public map and connect an optional self-hosted valuation API."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist"
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web", OUT)
host = ROOT / "deploy/public-host.json"
base = json.loads(host.read_text()).get("public_url") if host.exists() else None
if base and not base.startswith("https://"):
    raise ValueError("The public API must use HTTPS")
config = {"apiEnabled": bool(base), "apiBase": base or ""}
(OUT / "config.js").write_text("window.SG_HOMES_CONFIG = " + json.dumps(config) + ";\n")
(OUT / ".nojekyll").touch()
for name, key in [("blocks", "blocks"), ("condo", "projects"), ("landed", "projects")]:
    path = OUT / "data" / f"{name}.json"
    data = json.loads(path.read_text())
    assert data[key], f"No homes in {path}"
    print(f"{name}: {len(data[key]):,} homes, valuation month {data['meta']['valuation_month']}")
print(f"Public site ready: {OUT}")
