"""Download raw datasets from data.gov.sg into data/raw/."""
import sys
import time
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
API = "https://api-open.data.gov.sg/v1/public/api/datasets/{id}/poll-download"
INITIATE = "https://api-open.data.gov.sg/v1/public/api/datasets/{id}/initiate-download"

DATASETS = {
    "resale_2017_onwards.csv": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "resale_2015_2016.csv": "d_ea9ed51da2787afaf8e51f827c304208",
    "resale_2012_2014.csv": "d_2d5ff9ea31397b66239f245f57751537",
    "hdb_property_info.csv": "d_17f5382f26140b1fdae0ba2ef6239d2f",
    "hdb_buildings.geojson": "d_16b157c52ed637edd6ba1232e026258d",
    "mrt_exits.geojson": "d_b39d3a0871985372d7e1637193335da5",
    "schools.csv": "d_688b934f82c1059ed0a6993d2a829089",
}


def download_url(dataset_id: str) -> str:
    requests.get(INITIATE.format(id=dataset_id), timeout=30)
    for _ in range(30):
        r = requests.get(API.format(id=dataset_id), timeout=30).json()
        url = (r.get("data") or {}).get("url")
        if url:
            return url
        time.sleep(2)
    raise RuntimeError(f"no download url for {dataset_id}: {r}")


def main(names: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, dataset_id in DATASETS.items():
        if names and name not in names:
            continue
        out = RAW / name
        if out.exists():
            print(f"skip {name} (exists)")
            continue
        url = download_url(dataset_id)
        tmp = out.with_suffix(out.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(out)
        print(f"ok {name} {out.stat().st_size / 1e6:.1f} MB")
        time.sleep(3)  # data.gov.sg rate-limits unauthenticated calls


if __name__ == "__main__":
    main(sys.argv[1:])
