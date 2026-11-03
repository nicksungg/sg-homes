# sg-homes: Singapore home value map

Personal project: Zillow-style value estimates for HDB flats, condos and landed homes in Singapore, on a 3D map.

```
./refresh.sh                                          # download, geocode, train, forecast, score (~15 min)
cd pipeline && ../.venv/bin/python server.py 8765     # open http://localhost:8765
```

The server loads models at startup, so restart it after a refresh. A plain static server
(`python3 -m http.server` in `web/`) also works, minus the unit valuation forms.

Private-home data needs a free URA Data Service key in `.env` (`URA_ACCESS_KEY=...`), from
https://eservice.ura.gov.sg/maps/api/reg.html.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Download | `pipeline/download.py` | HDB resale 2012+, HDB property info, HDB building footprints, MRT exits, schools (all data.gov.sg) |
| Geocode | `pipeline/geocode.py` | Matches street names to footprint street codes by block-number overlap (99.3% of blocks); OneMap search fallback, cached in `data/raw/onemap_cache.json` |
| Features | `pipeline/features.py` | Lease age, storey, MRT/school/CBD distance, lagged market $/sqm index, lagged block and town comparables |
| Train | `pipeline/train.py` | `train_and_save()`: LightGBM on log($/sqm / market level); p10/p90 quantile models with conformal calibration; forward-in-time backtest. Shared by HDB, condo and landed |
| Forecast | `pipeline/forecast.py` | HDB lease-decay curve fit within blocks (block fixed effects, monotone spline, bootstrap band), market scenarios from history, 7.6-year backtest |
| Score | `pipeline/score.py` | Values a typical mid-floor unit of each flat type in each block for next month; writes `web/data/blocks.json` |
| URA download | `pipeline/ura_download.py` | All 4 batches of URA private transactions into a dated snapshot `data/raw/ura/<date>/`, merged into `data/raw/ura/archive.parquet` so history accumulates beyond URA's 5-year window |
| Private | `pipeline/private.py` | Condo and landed: URA SVY21 coordinates (OneMap fallback for 502 projects), mix-adjusted market index, lagged project/district comparables, train, export `web/data/{condo,landed}.json` |
| Serve | `pipeline/server.py` | Static map plus `/api/estimate` (HDB) and `/api/estimate_private` (condo/landed) unit valuation, using the batch scoring code path |
| UI | `web/index.html` | MapLibre + deck.gl: HDB footprints, condo/landed columns, layer toggles, search, flat-type and budget filters, detail panels. Deep links: `#hdb:<id>`, `#condo:<id>`, `#landed:<id>` |

## Backtests (model frozen, tested on later sales it never saw)

| | Test period | MAPE | Median error | Within 10% | 80% range covers |
|---|---|---|---|---|---|
| HDB | Jul 2025 – Oct 2026 (34k sales) | 3.9% | 2.9% | 94% | 75.5% |
| Condo | Nov 2025 – Oct 2026 (20k sales) | 4.8% | 3.5% | 90% | 79.6% |
| Landed | Nov 2025 – Oct 2026 (2.2k sales) | 11.8% | 8.7% | 56% | 77% |

Comparable-sales baselines: HDB 5.8%, condo 11.1%, landed 20.1% MAPE. Full reports: `models/*_eval_report.json`.

### Private-home notes
- **Mix-adjusted market index.** A median-$/sqm index swung ±10% month to month with the
  share of new launches (16–59%), which pushed condo MAPE to 7.6%. The index is now
  month effects from a within-project regression, refit each month on earlier sales only.
- **Floor premium** is constrained to be non-decreasing (LightGBM monotone constraint on the
  point model). Within projects the raw premium is about +0.3%/floor (resale) and +0.4%/floor
  (new sale). The model's response is weaker, and flat over some floor ranges.
- Condo estimates are per size band at the band's median floor, valued as a resale.
  Landed estimates are per house type at the median land area; plot shape, frontage and
  condition aren't in URA's data, hence the wide ranges.
- Projects are keyed by `project|street`. URA groups many landed estates as
  "LANDED HOUSING DEVELOPMENT", so a long road gets a single point.
- Demolished (en-bloc) projects are kept for training but not shown on the map.

## HDB projections
Future value = today's estimate × lease-decay ratio × market scenario. The decay curve is
learned from how each block's price relative to the market changes as it ages, so it's
the defensible part. The market path is a scenario (cooling / flat / long-run / boom,
taken from 2012–2026 history), not a prediction.

Lease-decay backtest (fit on pre-2019 sales, predict each block/flat type's relative
change from 2017–18 to 2024–26, 4,590 pairs, 7.6-year horizon): error 7.6% vs 9.3% for
"no change", correlation 0.57. By starting age the predicted vs actual relative change
matches within ~2 points (e.g. 40–50 yrs: −10.2% predicted, −10.9% actual).

Beyond the oldest flats sold so far (~60 years) there is no data. The curve then
assumes a straight line to $0 at 99 years (shown dashed). SERS/VERS are not modelled.

## Known gaps
- No projections for private homes yet (only 5 years of URA history).
- MRT distance uses today's network for old sales (stations that opened later count as present).
- Unit valuation only covers flat/house types that have sold in that block or project.
- The decay band covers curve uncertainty only, not valuation error or market risk.

## Public website and local hosting

Website: https://nicksungg.github.io/sg-homes/
Full self-hosted site and API: https://sg-homes.tail90d697.ts.net/
Local managed server: http://localhost:8767/

The GitHub Pages map calls the valuation API on this computer for custom size/floor
estimates. `deploy/public-host.json` supplies the public HTTPS address when the
Pages artifact is built. `web/config.js` uses the same-origin API for the local
and self-hosted editions. The computer must remain awake and online for custom
valuations. The GitHub map and existing estimates remain available during outages.

The Python server uses Waitress, bounds model prediction threads, serializes
inference, and only serves the map's five public files. The URA key, raw archives,
model files, Tailscale credentials and virtual environment are not published.
Install Python dependencies with `.venv/bin/python -m pip install -r requirements.txt`.

### Services on this computer

- `sg-homes.service`: map and valuation server, listening on loopback port 8767.
- `sg-homes-tailscale.service`: separate userspace Tailscale node named `sg-homes`,
  signed into the website owner's account, preserving the existing system VPN.
- `sg-homes-health.timer`: every two minutes calls a real HDB valuation, recovers
  a failed local server, and separately checks the public valuation endpoint.

Units are installed from `deploy/` into `~/.config/systemd/user/` and enabled.
User lingering is enabled, so they start at boot without an interactive login.
The Funnel setting persists in the dedicated node's private state directory.
Its authentication and certificates live under `~/.local/state/sg-homes-tailscale/`,
not in this repository. Tailscale may require account reauthorization when the
node key expires; the public monitor records that as unreachable.

```sh
systemctl --user status sg-homes.service sg-homes-tailscale.service
systemctl --user restart sg-homes.service
systemctl --user list-timers sg-homes-health.timer
journalctl --user -u sg-homes.service -u sg-homes-health.service -n 50
cat data/logs/hosting-health.json
tailscale --socket=/run/user/1000/sg-homes-tailscale/tailscaled.sock funnel status
```

After a data refresh, restart `sg-homes.service` to load the updated models.
To stop public access while keeping the local server running:

```sh
tailscale --socket=/run/user/1000/sg-homes-tailscale/tailscaled.sock funnel --https=443 off
```

### Publishing updates

Pushes to `main` deploy through `.github/workflows/pages.yml`. To preview the
same artifact, run `python3 scripts/build-pages.py`, then
`python3 -m http.server 8766 --directory dist`.

To update public estimates, run `./refresh.sh` locally, inspect its outputs,
restart the managed server, then commit and push the updated `web/data/` files.
Publishing does not retrain models or fetch new transactions automatically.
