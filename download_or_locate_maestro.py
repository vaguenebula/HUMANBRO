"""Find (and if needed extract or download) the MAESTRO MIDI dataset.

Usage:
    python download_or_locate_maestro.py --maestro_dir data/maestro-v3.0.0
    python download_or_locate_maestro.py --zip maestro-v3.0.0-midi.zip --extract_to data
    python download_or_locate_maestro.py --download --extract_to data

``locate_maestro`` accepts a dataset directory, a parent directory, or the
official ``maestro-v3.0.0-midi.zip`` and returns the directory that holds
``maestro-v*.csv`` (the metadata file with the official train/validation/test split).
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import setup_logging

log = logging.getLogger(__name__)

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def _find_csv(root: Path) -> Path | None:
    for pattern in ("maestro-v*.csv", "*/maestro-v*.csv"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[-1]
    return None


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for name in tqdm(members, desc=f"extracting {zip_path.name}", unit="file"):
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())):  # zip-slip guard
                raise RuntimeError(f"unsafe path in zip: {name}")
            if not target.exists():
                zf.extract(name, dest)
    csv = _find_csv(dest)
    if csv is None:
        raise FileNotFoundError(f"no maestro-v*.csv found after extracting {zip_path}")
    return csv.parent


def download(dest: Path, url: str = MAESTRO_URL) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / url.rsplit("/", 1)[-1]
    if out.exists():
        log.info("already downloaded: %s", out)
        return out
    log.info("downloading %s", url)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(out)
    return out


def locate_maestro(path: str | Path, extract_to: str | Path | None = None) -> Path:
    """Directory containing maestro-v*.csv; extracts a zip if given one."""
    path = Path(path)
    if path.is_file() and path.suffix == ".zip":
        dest = Path(extract_to) if extract_to else path.parent / "data"
        csv = _find_csv(dest) if dest.exists() else None
        return csv.parent if csv else extract_zip(path, dest)
    if path.is_dir():
        csv = _find_csv(path)
        if csv:
            return csv.parent
        zips = sorted(path.glob("maestro-v*-midi.zip"))
        if zips:
            return locate_maestro(zips[-1], extract_to or path)
    raise FileNotFoundError(
        f"MAESTRO not found at '{path}'. Pass the extracted dataset directory or the "
        f"maestro-v3.0.0-midi.zip file, or run with --download."
    )


def read_metadata(root: Path) -> pd.DataFrame:
    csv = _find_csv(root)
    if csv is None:
        raise FileNotFoundError(f"no maestro-v*.csv in {root}")
    meta = pd.read_csv(csv)
    required = {"midi_filename", "split", "canonical_composer", "canonical_title"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"{csv} lacks columns {sorted(missing)}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maestro_dir", default=None, help="extracted dataset directory (or its parent)")
    ap.add_argument("--zip", default=None, help="path to maestro-v3.0.0-midi.zip")
    ap.add_argument("--download", action="store_true", help=f"download from {MAESTRO_URL}")
    ap.add_argument("--extract_to", default="data", help="where to extract zips (default: data)")
    args = ap.parse_args()
    setup_logging()

    if args.download:
        root = locate_maestro(download(Path(args.extract_to)), args.extract_to)
    elif args.zip:
        root = locate_maestro(args.zip, args.extract_to)
    elif args.maestro_dir:
        root = locate_maestro(args.maestro_dir, args.extract_to)
    else:
        ap.error("give --maestro_dir, --zip or --download")

    meta = read_metadata(root)
    present = meta["midi_filename"].map(lambda f: (root / f).exists())
    log.info("MAESTRO root: %s", root)
    log.info("%d performances listed, %d MIDI files present", len(meta), int(present.sum()))
    for split, grp in meta.groupby("split"):
        log.info("  %-10s %4d files  %6.1f hours", split, len(grp), grp["duration"].sum() / 3600)


if __name__ == "__main__":
    main()
