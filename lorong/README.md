# sg-home.com: demo property portal (code name Lorong)

An original Singapore property portal built on [sg-homes](https://github.com/nicksungg/sg-homes):
the usual listing-site features, plus the sg-homes 3D value map, home valuations and lease
outlook built in, and an independent estimate next to every asking price.

**Everything here is demo data.** Listings, agents, agencies, rents and photos are synthetic.
Each listing sits on a real HDB block or URA project, and its estimate comes from models
trained on public HDB resale and URA transaction data.

Lorong lives on the `lorong` branch of sg-homes, in this `lorong/` folder. It needs a refreshed
sg-homes checkout (models, `web/data/*.json`, processed parquet files and the `.venv`), which it
finds via `SG_HOMES_DIR`, else the repo root, else a sibling `sg-homes` folder.

```
export SG_HOMES_DIR=~/Desktop/nicksung/sg-homes
$SG_HOMES_DIR/.venv/bin/python generate.py      # build data/*.json from sg-homes (~20 s)
$SG_HOMES_DIR/.venv/bin/python app.py 8780      # open http://localhost:8780
```

## Hosting
Public site: https://sg-homes.tail90d697.ts.net:8443 (local: http://localhost:8780).

| Piece | What it does |
|---|---|
| `deploy/lorong.service` | systemd user service running `app.py` on 127.0.0.1:8780, restarts on failure |
| sg-homes' dedicated Tailscale node | Funnel on HTTPS port 8443 → 127.0.0.1:8780 (sg-homes itself stays on 443) |
| `deploy/lorong-health.{service,timer}` + `scripts/check_host.py` | every 2 minutes: loads the page and listings and runs a real valuation, locally and through the public URL; restarts the service if the local check fails; status in `data/logs/health.json` |

Install or update: copy `deploy/lorong*.{service,timer}` to `~/.config/systemd/user/`, then
`systemctl --user daemon-reload && systemctl --user enable --now lorong.service lorong-health.timer`.
Publish: `tailscale --socket=$XDG_RUNTIME_DIR/sg-homes-tailscale/tailscaled.sock funnel --bg --https=8443 http://127.0.0.1:8780`.
After an sg-homes refresh: rerun `generate.py`, then `systemctl --user restart lorong.service`.

Public-facing safeguards in `app.py`: enquiries are only visible to the browser that sent them,
owner emails are stored but never returned, POSTs are rate-limited per visitor (30 per 10 min)
and capped overall, valuations share sg-homes' inference lock, request bodies are size-limited,
and only `web/`, the generated JSON and the three sg-homes map files are served.

## Features
| Area | What's there |
|---|---|
| Home | Live 3D city of all 13.5k valued blocks/projects (height = storeys, colour = estimated $ psf), search, below-estimate deals, HDB town explorer, backtest highlights |
| Value map (`#/map`) | The sg-homes map inside Lorong: HDB footprints and condo/landed columns coloured by value, search, flat-type and budget filters, block panels with estimates, unit valuation, lease outlook chart, recent sales and the homes listed there. Deep link `#/map?focus=hdb:<id>` |
| What's it worth? (`#/value`) | Pick a block/project, floor and size; get an estimate, range, recent sales, lease outlook; one click to list it (prefills the sell form) |
| How we value (`#/about`) | Backtest results for each model and the lease-decay curve, read from the sg-homes eval reports |
| Search | Buy/rent, keyword, property type, HDB town or district, price, bedrooms, size, tenure, floor level, MRT distance, listing age, furnishing, "below estimate only"; 6 sort orders; paging; list or map view with clustering and "Search this area" |
| Listing page | Placeholder photo gallery and lightbox, key facts, details, facilities, location map, **estimate vs asking** with 80% range, real recent transactions at that block/project, HDB lease-decay outlook, mortgage and stamp-duty calculators, agent contact form, similar homes nearby |
| Shortlist & compare | Save homes (♡), saved searches with live match counts, side-by-side comparison of up to 4 homes (stored in the browser) |
| Condo directory | ~2,300 projects with filters, new-launch view, project pages with estimates per size band, $ psf scatter and transaction history |
| Market trends | Median $ psf charts (HDB by type, condo new vs resale, condo/landed by region, volumes) with hover |
| Calculators | Affordability (TDSR/MSR), mortgage repayment, stamp duty (BSD + ABSD) using 2023 rules |
| List a home | Pick a real block/project, get a price suggestion from the model, publish (stored in SQLite) |
| Enquiries | Contact forms save to SQLite; `#/enquiries` lists what was sent |

## Files
- `generate.py`: synthetic listings, agents, condo directory, trend series, hero-city points (`hero.json`) and HDB town summaries (`areas.json`) from sg-homes; backtest stats go into `listings.json` meta. Valuations go through sg-homes' `estimate_hdb` / `estimate_private`, the same functions its API uses.
- `app.py`: stdlib HTTP server. Serves `web/`, `data/*.json` and sg-homes' map data at `/sg/{blocks,condo,landed}.json`, plus `/api/ref`, `/api/forecast`, `/api/places`, `/api/value`, `/api/listings`, `/api/user-listings`, `/api/enquiries`.
- `web/`: `index.html`, `styles.css`, `app.js` (hash-routed single page, MapLibre + deck.gl maps, SVG charts and drawn placeholder photos). Fonts: Fraunces and Inter from Google Fonts.
- `data/lorong.db`: created on first run; delete it to reset posted listings and enquiries.

## Notes
- Asking prices are the estimate × a random seller markup (median +3.4%; about a quarter are below estimate), so the "below estimate" feature has something to find. Real asking prices aren't used.
- Rents come from a gross-yield assumption; there's no rent model behind them.
- Calculator rules (BSD/ABSD rates, 75% LTV, TDSR 55%, MSR 30%, 4%/3% stress rates) reflect 2023 regulations. Check IRAS, MAS and HDB for current rules.
- Nothing is taken from any other property portal: no scraping, no copied content or design.
