"""
e6_tag_utils.py  –  Download, cache, and sanitise e621 DB-export tags
"""

from __future__ import annotations
import gzip, shutil, sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ───────────────────────── constants ─────────────────────────────
EXPORT_URL      = "https://e621.net/db_export/"
# ↓↓↓ corrected prefixes ↓↓↓
CSV_PREFIXES    = ("tags-", "tag_aliases-", "tag_implications-")
UNWANTED_CAT_ID = {1, 3, 4, 6, 7}        # artist, copyright, character, invalid, lore
YEAR_PATTERN    = r"^(19|20)\d{2}$"
CACHE_DIR       = Path(__file__).parent / "e6_cache"
CACHE_DIR.mkdir(exist_ok=True)
MAX_LOOKBACK    = 90                     # days


# ───────────────────── helper: HTTP session ──────────────────────
def _http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "tag-cleaner/0.2 (+github.com/yourname)"})
    return s

# ─────────── download + extract with tqdm byte progress ──────────
def _download_csv(prefix: str, day: datetime.date, sess: requests.Session) -> Path | None:
    name_gz  = f"{prefix}{day:%Y-%m-%d}.csv.gz"
    url      = EXPORT_URL + name_gz
    gz_path  = CACHE_DIR / name_gz
    csv_path = gz_path.with_suffix("")   # strip .gz

    if csv_path.exists():          # already on disk & extracted
        return csv_path

    r = sess.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        return None

    total = int(r.headers.get("content-length", 0))
    bar   = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {name_gz}")
    with gz_path.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            bar.update(len(chunk))
    bar.close()

    # gunzip →
    with gzip.open(gz_path, "rb") as src, csv_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)
    return csv_path

# ─────────────────── ensure we have a “fresh” CSV ─────────────────
def _ensure_csv(prefix: str) -> Path:
    sess   = _http_session()
    today  = datetime.utcnow().date()
    newest = None

    # find newest on disk
    for p in sorted(CACHE_DIR.glob(f"{prefix}*.csv"), reverse=True):
        newest = p
        break

    # if newest ≤ MAX_LOOKBACK days old → good enough
    if newest:
        age = (today - datetime.strptime(p.stem[len(prefix):], "%Y-%m-%d").date()).days
        if age < MAX_LOOKBACK:
            return newest

    # otherwise attempt downloads: today → today-MAX_LOOKBACK
    for delta in range(MAX_LOOKBACK + 1):
        csv = _download_csv(prefix, today - timedelta(days=delta), sess)
        if csv:
            newest = csv
            break

    if newest:
        # purge older same-prefix files
        for old in CACHE_DIR.glob(f"{prefix}*.csv"):
            if old != newest:
                old.unlink(missing_ok=True)
        return newest

    # fall-back: no download but an old one exists
    if newest:
        print(f"[e6] ⚠  Using stale {newest.name} (no recent export found).", file=sys.stderr)
        return newest

    raise RuntimeError(f"[e6] ❌  Could not obtain {prefix}<date>.csv within {MAX_LOOKBACK} days.")


def _bootstrap_exports() -> Tuple[Path, Path, Path]:
    # show a neat 3-step bar
    for prefix in CSV_PREFIXES:
        print(f"[e6] Checking {prefix}…")
    return tuple(_ensure_csv(pfx) for pfx in CSV_PREFIXES)   # type: ignore

# ─────────────────────── load into memory ────────────────────────
def _load_data() -> Tuple[Dict[str, str], Set[str], Dict[str, int]]:
    tags_csv, aliases_csv, impl_csv = _bootstrap_exports()

    bar = tqdm(total=3, desc="Loading CSVs", unit="file")
    tags_df = pd.read_csv(tags_csv, usecols=["name", "category", "post_count"])
    bar.update()

    # aliases – accepts either col style
    alias_df = pd.read_csv(
        aliases_csv,
        usecols=lambda c: c in {"alias_name", "antecedent_name", "consequent_name"},
    )
    bar.update()

    impl_df  = pd.read_csv(
        impl_csv,
        usecols=lambda c: c in {"antecedent_name", "consequent_name"},
    )
    bar.update(); bar.close()

    tags_df  = tags_df[tags_df["post_count"] > 0]

    # ─ merge alias + implication antecedent → consequent
    alias_df = alias_df.rename(columns={"alias_name": "antecedent_name"})
    mapping  = dict(zip(alias_df["antecedent_name"].astype(str),
                        alias_df["consequent_name"].astype(str)))
    mapping.update(
        zip(impl_df["antecedent_name"].astype(str),
            impl_df["consequent_name"].astype(str))
    )

    valid_tags = set(tags_df["name"].astype(str))
    tag_cats   = dict(zip(tags_df["name"].astype(str), tags_df["category"].astype(int)))
    return mapping, valid_tags, tag_cats


_ALIAS_MAP, _VALID_SET, _CATS = _load_data()

# ───────────────────── public sanitiser ──────────────────────────
def sanitize_tag_list(tags: list[str]) -> list[str]:
    """
    Four-pass pipeline (all deduplicated, order-preserving):
      1. alias / implication  → canonical tag
      2. drop tags not present in tags.csv
      3. drop YYYY tags
      4. drop unwanted category IDs (artist, character, copyright, lore, invalid)
    """
    out, seen = [], set()
    for tag in tags:
        tag = _ALIAS_MAP.get(tag, tag)          # ① replace if alias/implication
        if tag not in _VALID_SET:               # ② unknown → skip
            continue
        if any(char.isdigit() for char in tag) and len(tag) == 4:   # crude year check
            continue
        if _CATS.get(tag) in UNWANTED_CAT_ID:   # ④ category filter
            continue
        if tag not in seen:
            out.append(tag); seen.add(tag)
    return out
