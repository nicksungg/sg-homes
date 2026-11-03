"""Lorong server: static site, demo data, and APIs backed by sg-homes.

GET  /sg/{blocks,condo,landed}.json            the sg-homes value map data, slimmed for the map
GET  /api/health                              liveness + data/model check
GET  /api/ref?kind=hdb|condo|landed&id=<id>   real block/project record (estimates, transactions)
GET  /api/forecast                            HDB lease-decay curve + market scenarios
GET  /api/places?q=<text>&kind=<kind>         search real blocks/projects (for "Sell")
POST /api/value                               value a unit (same functions as sg-homes' API)
POST /api/listings                            post a listing (validated, valued, stored)
GET  /api/user-listings                       listings posted through the site (no owner emails)
POST /api/enquiries                           send an enquiry to an agent
GET  /api/enquiries?client=<id>               enquiries sent from that browser only

The sg-homes checkout (models, map data, pipeline code) is found via SG_HOMES_DIR, else
the repo root this folder sits in, else a sibling "sg-homes" folder.
Run with the sg-homes venv: $SG_HOMES_DIR/.venv/bin/python app.py [port]
"""
import gzip
import json
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import date
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


def find_sg_homes() -> Path:
    candidates = [os.environ.get("SG_HOMES_DIR"), ROOT.parent, ROOT.parent.parent / "sg-homes"]
    for c in candidates:
        if c and (Path(c) / "models" / "hdb_serving.json").exists():
            return Path(c).resolve()
    raise SystemExit("sg-homes models not found; set SG_HOMES_DIR to the sg-homes checkout")


SG = find_sg_homes()
sys.path.insert(0, str(SG / "pipeline"))
import server as sg  # noqa: E402  (sg-homes valuation API functions)

SQFT = 10.7639
DB_PATH = ROOT / "data" / "lorong.db"
DB_LOCK = threading.Lock()
# Valuations share sg-homes' lock so the two sites never run inference concurrently.
INFERENCE_LOCK = getattr(sg, "INFERENCE_LOCK", threading.Lock())
MAX_USER_LISTINGS, MAX_ENQUIRIES = 500, 10_000
RATE_LIMIT = (30, 600)  # POSTs per IP per 10 minutes
MAX_BODY = 100_000
REFS = {
    "hdb": {b["id"]: b for b in json.loads((SG / "web/data/blocks.json").read_text())["blocks"]},
    **{seg: {p["id"]: p for p in json.loads((SG / f"web/data/{seg}.json").read_text())["projects"]}
       for seg in ["condo", "landed"]},
}
for b in REFS["hdb"].values():
    b.pop("poly", None)  # footprints aren't needed here
FORECAST = json.loads((SG / "models/forecast.json").read_text())
DATA_FILES = {"listings.json", "agents.json", "projects.json", "trends.json", "hero.json", "areas.json"}
SG_FILES = {"blocks.json", "condo.json", "landed.json"}  # the sg-homes value map data, served at /sg/


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS enquiries (
            id INTEGER PRIMARY KEY, listing TEXT NOT NULL, agent TEXT, name TEXT NOT NULL,
            email TEXT NOT NULL, phone TEXT, message TEXT NOT NULL, created TEXT NOT NULL, client TEXT);
        CREATE TABLE IF NOT EXISTS user_listings (
            id TEXT PRIMARY KEY, body TEXT NOT NULL, created TEXT NOT NULL, owner_email TEXT);
        """)
        # Databases created before the public deployment lack these columns.
        for table, col in [("enquiries", "client"), ("user_listings", "owner_email")]:
            if col not in [r["name"] for r in con.execute(f"PRAGMA table_info({table})")]:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def place_label(kind: str, x: dict) -> str:
    if kind == "hdb":
        return f"Blk {x['block']} {x['street'].title()} ({x['town'].title()})"
    name = f"Landed homes, {x['street'].title()}" if x["project"] == "LANDED HOUSING DEVELOPMENT" else x["project"].title()
    return f"{name}, {x['street'].title()} (D{x['district']})"


def places(q: dict) -> tuple[int, dict]:
    kind, text = q.get("kind", "hdb"), q.get("q", "").upper().replace("BLK", "").split()
    if kind not in REFS or not text:
        return 400, {"error": "need kind and q"}
    out = []
    for x in REFS[kind].values():
        hay = f"{x.get('block', '')} {x.get('project', '')} {x['street']} {x.get('town', '')}"
        if all(t in hay for t in text):
            out.append({"id": x["id"], "label": place_label(kind, x),
                        "units": [{"type": u["type"], "sqm": u["sqm"]} for u in x["units"]],
                        "floors": x.get("floors")})
            if len(out) == 10:
                break
    return 200, {"results": out}


def value(body: dict) -> tuple[int, dict]:
    kind = body.get("kind")
    params = {k: str(v) for k, v in body.items() if v is not None and k != "kind"}
    if kind not in ("hdb", "condo", "landed"):
        return 400, {"error": "kind must be hdb, condo or landed"}
    if not INFERENCE_LOCK.acquire(timeout=5):
        return 503, {"error": "The valuation server is busy. Please try again shortly."}
    try:
        return sg.estimate_hdb(params) if kind == "hdb" else sg.estimate_private({**params, "seg": kind})
    except Exception as err:  # noqa: BLE001 - never leak a traceback to the browser
        print(f"valuation failed for {params}: {err!r}", flush=True)
        return 500, {"error": "Unable to value this home. Please try again."}
    finally:
        INFERENCE_LOCK.release()


def post_listing(body: dict) -> tuple[int, dict]:
    required = ["kind", "id", "mode", "price", "sqm", "beds", "baths", "name", "email"]
    missing = [k for k in required if body.get(k) in (None, "")]
    if missing:
        return 400, {"error": f"missing: {', '.join(missing)}"}
    try:
        price, sqm, beds, baths = int(body["price"]), float(body["sqm"]), int(body["beds"]), int(body["baths"])
    except (TypeError, ValueError):
        return 400, {"error": "price, sqm, beds and baths must be numbers"}
    if body["mode"] not in ("sale", "rent") or not 0 < price < 1e9 or not 0 < beds < 12 or not 0 < baths < 12:
        return 400, {"error": "check mode, price, bedrooms and bathrooms"}
    if "@" not in str(body["email"]):
        return 400, {"error": "email looks invalid"}
    kind, ref = body["kind"], REFS.get(body["kind"], {}).get(body["id"])
    if not ref:
        return 404, {"error": "unknown block or project"}
    vparams = {"kind": kind, "id": body["id"], "sqm": sqm}
    if kind == "hdb":
        vparams |= {"type": body.get("type"), "storey": body.get("floor")}
    elif kind == "condo":
        vparams |= {"floor": body.get("floor")}
    else:
        vparams |= {"type": body.get("type")}
    status, val = value(vparams)
    if status != 200:
        return status, val
    with DB_LOCK, db() as con:
        n = con.execute("SELECT COUNT(*) FROM user_listings").fetchone()[0]
        if n >= MAX_USER_LISTINGS:
            return 429, {"error": "This demo has reached its limit of owner listings."}
        lid = f"U{n + 1:05d}"
        listing = {
            "id": lid, "kind": kind, "mode": body["mode"], "user": True,
            "name": place_label(kind, ref).split(" (")[0], "address": place_label(kind, ref).split(" (")[0],
            "property_type": ("HDB " + str(body["type"]).title()) if kind == "hdb" else (body.get("type") or ref["types"][0]),
            "street": ref["street"], "town": ref.get("town", "").title() or None, "district": ref.get("district"),
            "segment": ref.get("segment"), "lat": ref["lat"], "lon": ref["lon"],
            "sqm": sqm, "sqft": int(round(sqm * SQFT)), "beds": beds, "baths": baths,
            "floor": int(float(body["floor"])) if body.get("floor") else None, "floor_label": None, "floors": ref.get("floors"),
            "tenure": "99-year leasehold" if kind == "hdb" else ref["tenure"].split(" lease")[0],
            "lease_start": ref.get("lease_start"), "remaining_lease": None,
            "mrt": ref["mrt"], "mrt_m": ref["mrt_m"], "schools_1km": ref["schools_1km"],
            "ref": {"kind": kind, "id": body["id"], "unit": body.get("type")}, "valuation": val,
            "price": price, "psf": round(price / (sqm * SQFT), 2), "furnishing": str(body.get("furnishing") or "Unfurnished")[:40],
            "facing": "–", "listed": date.today().isoformat(), "agent": None,
            "owner": {"name": str(body["name"]).strip()[:100]},  # email is stored privately, never returned
            "images": [{"room": r, "seed": hash((lid, r)) % 10_000} for r in ["Living room", "Kitchen", "Bedroom", "Bathroom"]],
            "description": str(body.get("description") or "").strip()[:2000] or "Listed by owner.",
            "features": [], "facilities": [], "views": 0,
        }
        con.execute("INSERT INTO user_listings (id, body, created, owner_email) VALUES (?, ?, ?, ?)",
                    (lid, json.dumps(listing), listing["listed"], str(body["email"]).strip()[:200]))
    return 201, listing


def user_listings(_: dict) -> tuple[int, dict]:
    with db() as con:
        rows = con.execute("SELECT body FROM user_listings ORDER BY created DESC, id DESC").fetchall()
    listings = [json.loads(r["body"]) for r in rows]
    for l in listings:  # rows written before the public deployment carried the owner's email
        if l.get("owner"):
            l["owner"] = {"name": l["owner"].get("name", "Owner")}
    return 200, {"listings": listings}


def post_enquiry(body: dict) -> tuple[int, dict]:
    missing = [k for k in ["listing", "name", "email", "message", "client"] if not str(body.get(k, "")).strip()]
    if missing:
        return 400, {"error": f"missing: {', '.join(missing)}"}
    if "@" not in str(body["email"]):
        return 400, {"error": "email looks invalid"}
    with DB_LOCK, db() as con:
        if con.execute("SELECT COUNT(*) FROM enquiries").fetchone()[0] >= MAX_ENQUIRIES:
            return 429, {"error": "This demo has reached its limit of enquiries."}
        cur = con.execute("INSERT INTO enquiries (listing, agent, name, email, phone, message, created, client) "
                          "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)",
                          (str(body["listing"])[:20], str(body.get("agent") or "")[:20] or None, str(body["name"]).strip()[:100],
                           str(body["email"]).strip()[:200], str(body.get("phone", "")).strip()[:40],
                           str(body["message"]).strip()[:2000], str(body["client"])[:64]))
    return 201, {"id": cur.lastrowid}


def enquiries(q: dict) -> tuple[int, dict]:
    client = q.get("client", "")
    if len(client) < 16:
        return 400, {"error": "need client"}
    with db() as con:
        rows = con.execute("SELECT id, listing, agent, name, message, created FROM enquiries WHERE client = ? "
                           "ORDER BY id DESC", (client,)).fetchall()
    return 200, {"enquiries": [dict(r) for r in rows]}


def ref(q: dict) -> tuple[int, dict]:
    x = REFS.get(q.get("kind"), {}).get(q.get("id"))
    return (200, x) if x else (404, {"error": "not found"})


def health(_: dict) -> tuple[int, dict]:
    ok = all((ROOT / "data" / f).exists() for f in DATA_FILES)
    return (200 if ok else 503), {"status": "ok" if ok else "missing data", "sg_homes": str(SG),
                                  "models": list(getattr(sg, "VALUERS", {}))}


GET = {"/api/ref": ref, "/api/forecast": lambda q: (200, FORECAST), "/api/places": places,
       "/api/user-listings": user_listings, "/api/enquiries": enquiries, "/api/health": health}
POST = {"/api/value": value, "/api/listings": post_listing, "/api/enquiries": post_enquiry}

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()
_gz_cache: dict[Path, tuple[float, bytes, bytes, str]] = {}
_gz_lock = threading.Lock()


def rate_limited(ip: str) -> bool:
    limit, window = RATE_LIMIT
    now = time.monotonic()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False


def simplify(pts: list[tuple[int, int]], tol: float) -> list[tuple[int, int]]:
    """Douglas-Peucker: keep only the corners that move the outline by more than tol."""
    if len(pts) < 3:
        return pts
    (ax, ay), (bx, by) = pts[0], pts[-1]
    dx, dy = bx - ax, by - ay
    norm = (dx * dx + dy * dy) ** 0.5
    dist = [abs(dy * (x - ax) - dx * (y - ay)) / norm if norm else ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5
            for x, y in pts[1:-1]]
    i = max(range(len(dist)), key=dist.__getitem__)
    if dist[i] <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:i + 2], tol)[:-1] + simplify(pts[i + 1:], tol)


def encode_ring(ring: list) -> list[int]:
    """Footprint ring as integers at 1e-5 degrees (~1 m), simplified to ~1 m: first point, then deltas."""
    pts = [(round(lon * 1e5), round(lat * 1e5)) for lon, lat in ring]
    pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
    pts = simplify(pts, 1.0) if len(pts) > 4 else pts
    out, px, py = [], 0, 0
    for x, y in pts:
        out += [x - px, y - py]
        px, py = x, y
    return out


def slim_map_data(path: Path) -> bytes:
    """The value map draws from estimates and footprints only. Transactions (fetched per place
    through /api/ref) and full-precision footprints would make the download ~6x larger."""
    data = json.loads(path.read_bytes())
    for x in data.get("blocks", data.get("projects", [])):
        x.pop("tx", None)
        if x.get("poly"):
            x["pz"] = [encode_ring(r) for r in x.pop("poly")]
    return json.dumps(data, separators=(",", ":")).encode()


def file_bytes(path: Path, transform=None) -> tuple[bytes, bytes, str]:
    """Raw and gzipped file contents plus an ETag, cached until the file changes."""
    mtime = path.stat().st_mtime
    with _gz_lock:
        hit = _gz_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1:]
    raw = transform(path) if transform else path.read_bytes()
    gz = gzip.compress(raw, compresslevel=9)
    etag = f'"{int(mtime)}-{len(raw)}"'
    with _gz_lock:
        _gz_cache[path] = (mtime, raw, gz, etag)
    return raw, gz, etag


class Handler(SimpleHTTPRequestHandler):
    timeout = 30  # drop stalled connections

    def send_header(self, keyword: str, value: str) -> None:
        if keyword.lower() == "cache-control":
            self._cache_set = True
        super().send_header(keyword, value)

    def end_headers(self) -> None:
        # Make browsers revalidate pages and scripts so a redeploy shows up on the next load.
        if not getattr(self, "_cache_set", False):
            self.send_header("Cache-Control", "no-cache")
        self._cache_set = False
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()

    def client_ip(self) -> str:
        # Behind Tailscale Funnel every request arrives from loopback; the proxy adds X-Forwarded-For.
        peer = self.client_address[0]
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() if peer in ("127.0.0.1", "::1") and fwd else peer

    def send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in GET:
            return self.send_json(*GET[url.path]({k: v[0] for k, v in parse_qs(url.query).items()}))
        if url.path.startswith(("/data/", "/sg/")):
            folder, name = url.path[1:].split("/", 1)
            allowed, base = (DATA_FILES, ROOT / "data") if folder == "data" else (SG_FILES, SG / "web" / "data")
            if name not in allowed:
                return self.send_json(404, {"error": "not found"})
            raw, gz, etag = file_bytes(base / name, slim_map_data if folder == "sg" else None)
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            use_gz = "gzip" in self.headers.get("Accept-Encoding", "")
            data = gz if use_gz else raw
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", etag)
            if use_gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path not in POST:
            return self.send_json(404, {"error": "not found"})
        if rate_limited(self.client_ip()):
            return self.send_json(429, {"error": "Too many requests. Please wait a few minutes."})
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self.send_json(400, {"error": "bad Content-Length"})
        if length > MAX_BODY:
            return self.send_json(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "invalid JSON"})
        if not isinstance(body, dict):
            return self.send_json(400, {"error": "expected a JSON object"})
        self.send_json(*POST[url.path](body))

    def log_message(self, fmt: str, *args) -> None:
        # Log API calls and errors only; static asset hits would flood the journal.
        if "/api/" in self.path or (args and str(args[1]).startswith(("4", "5"))):
            print(f"{self.client_ip()} {fmt % args}", flush=True)


if __name__ == "__main__":
    init_db()
    missing = [f for f in DATA_FILES if not (ROOT / "data" / f).exists()]
    if missing:
        raise SystemExit(f"missing data/{missing}; run generate.py first")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8780
    print(f"Lorong on http://localhost:{port} (sg-homes: {SG})", flush=True)
    # Build the slimmed map files now so the first visitor doesn't wait for them.
    threading.Thread(target=lambda: [file_bytes(SG / "web" / "data" / f, slim_map_data) for f in SG_FILES], daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "web"))).serve_forever()
