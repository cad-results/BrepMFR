"""Download STEP files referenced by GrabCAD library URLs in a Markdown file.

GrabCAD's public library exposes a JSON model endpoint and per-file download
endpoint. Downloads are gated by an authenticated session cookie. Paste your
`_grabcad_session=...` cookie value (with the `_grabcad_session=` prefix, or
just the raw value) into ``~/.grabcad_session`` before running this.

For every URL like ``https://grabcad.com/library/<slug>``:

1. Query ``https://grabcad.com/library/<slug>.json`` for the model metadata
2. Iterate ``files`` in the metadata, filter to STEP / STP entries
3. Hit ``https://grabcad.com/library/file/<file_id>/download`` for each STEP
4. Save the STEP into ``--out_dir`` using ``<slug>__<basename>.step``
5. Skip if the on-disk file already exists (idempotent resume)
6. Enforce a size ceiling with ``--max_size_mb``
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

LINK_RE = re.compile(r"https://grabcad\.com/library/([A-Za-z0-9_\-]+)")
STEP_EXTS = (".step", ".stp")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def load_session_cookie(path: Path) -> str:
    if not path.exists():
        raise SystemExit(
            f"Session cookie file not found at {path}. Paste your "
            "_grabcad_session cookie value into that file."
        )
    raw = path.read_text().strip()
    if raw.startswith("_grabcad_session="):
        raw = raw.split("=", 1)[1]
    if not raw:
        raise SystemExit(f"Empty session cookie in {path}")
    return raw


def extract_slugs(links_md: Path) -> list[str]:
    seen = set()
    slugs: list[str] = []
    for match in LINK_RE.finditer(links_md.read_text()):
        slug = match.group(1)
        if slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


def make_session(cookie_value: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    session.cookies.set("_grabcad_session", cookie_value, domain=".grabcad.com")
    return session


def fetch_metadata(session: requests.Session, slug: str) -> dict | None:
    url = f"https://grabcad.com/library/{slug}.json"
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def is_step_file(entry: dict) -> bool:
    name = (entry.get("filename") or entry.get("name") or "").lower()
    return name.endswith(STEP_EXTS)


def download_step_files(
    session: requests.Session,
    slug: str,
    out_dir: Path,
    max_bytes: int,
    log: logging.Logger,
) -> tuple[int, int]:
    metadata = fetch_metadata(session, slug)
    if not metadata:
        log.warning("[%s] metadata fetch failed", slug)
        return 0, 0

    files = metadata.get("files") or metadata.get("attachments") or []
    step_files = [f for f in files if is_step_file(f)]
    if not step_files:
        log.info("[%s] no STEP files in metadata", slug)
        return 0, 0

    downloaded = 0
    skipped = 0
    for entry in step_files:
        file_id = entry.get("id") or entry.get("file_id")
        filename = entry.get("filename") or entry.get("name") or f"{file_id}.step"
        if file_id is None:
            log.warning("[%s] entry missing id: %s", slug, filename)
            continue
        ext = Path(filename).suffix.lower()
        if ext not in STEP_EXTS:
            ext = ".step"
        stem = Path(filename).stem
        out_path = out_dir / f"{slug}__{stem}{ext}"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue

        dl_url = f"https://grabcad.com/library/file/{file_id}/download"
        try:
            resp = session.get(dl_url, stream=True, timeout=120, allow_redirects=True)
        except requests.RequestException as exc:
            log.warning("[%s] request failed for %s: %s", slug, filename, exc)
            continue

        if resp.status_code != 200:
            log.warning(
                "[%s] status %s for %s", slug, resp.status_code, filename
            )
            continue

        # Refuse to write giant STEP files (>max_bytes)
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > max_bytes:
            log.info("[%s] skipping %s (size %s > cap)", slug, filename, cl)
            continue

        tmp = out_path.with_suffix(out_path.suffix + ".part")
        try:
            written = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise IOError("size cap exceeded mid-stream")
                    fh.write(chunk)
            tmp.rename(out_path)
            downloaded += 1
            log.info("[%s] wrote %s (%d bytes)", slug, out_path.name, written)
        except (IOError, OSError) as exc:
            log.warning("[%s] failed writing %s: %s", slug, filename, exc)
            if tmp.exists():
                tmp.unlink()

    return downloaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links_md", type=Path, default=Path("LINKS.md"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("brepformer/data/hard_samples"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_size_mb", type=int, default=64)
    parser.add_argument(
        "--cookie_file",
        type=Path,
        default=Path.home() / ".grabcad_session",
    )
    parser.add_argument("--log_file", type=Path, default=Path("hard_samples_download.log"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("grabcad")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cookie = load_session_cookie(args.cookie_file)
    slugs = extract_slugs(args.links_md)
    log.info("Parsed %d unique slugs from %s", len(slugs), args.links_md)

    max_bytes = args.max_size_mb * 1024 * 1024

    total_dl = 0
    total_skip = 0

    def worker(slug: str) -> tuple[int, int]:
        session = make_session(cookie)
        return download_step_files(session, slug, args.out_dir, max_bytes, log)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for slug, result in zip(slugs, pool.map(worker, slugs)):
            dl, sk = result
            total_dl += dl
            total_skip += sk

    log.info("Done. downloaded=%d, skipped_existing=%d", total_dl, total_skip)


if __name__ == "__main__":
    main()
