"""Package the public map without Python APIs or private inputs."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist"
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web", OUT)
(OUT / "config.js").write_text("window.SG_HOMES_CONFIG = { apiEnabled: false };\n")
(OUT / ".nojekyll").touch()
for name, key in [("blocks", "blocks"), ("condo", "projects"), ("landed", "projects")]:
    path = OUT / "data" / f"{name}.json"
    data = json.loads(path.read_text())
    assert data[key], f"No homes in {path}"
    print(f"{name}: {len(data[key]):,} homes, valuation month {data['meta']['valuation_month']}")
print(f"Public site ready: {OUT}")
