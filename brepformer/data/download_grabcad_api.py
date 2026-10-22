"""GrabCAD downloader driven by the community REST API (Phase 2, second pass).

Why this exists
---------------
``download_grabcad_playwright.py`` drives the page UI: find the "Download
files" button, click it, wait for a browser download. Over 476 slugs that
kept ~80 and missed 396 — the log splits the misses into "no download button
found" and "no download fired". Probing the DOM (see
``probe_grabcad_dom.py`` / ``probe_grabcad_files.py``) showed the button is
almost always present and the session is fine; the flow itself is the
problem. "Download files" calls ``downloadArchive()``, which asks the server
to build a ZIP of *every* file in the model — for most models that either
never fires a client download event inside the timeout, or produces a
multi-hundred-MB archive of renderings we throw away.

The model page's own file rows point at
``/community/api/v1/models/<slug>/files/download?cadid=...``, and the sibling
``/community/api/v1/models/<slug>/files`` returns a JSON manifest listing
every file with its ``extension`` and ``download_url``. So instead:

1. Launch a headed persistent Chromium once — this is still required, because
   AWS WAF blocks plain HTTP clients. The browser solves the challenge and the
   WAF/session cookies live in the persistent profile.
2. Read the manifest through ``context.request`` (an APIRequestContext that
   shares those cookies but skips page rendering — roughly 10x faster than
   ``page.goto`` per slug).
3. Recurse into ``folders[]``, since many models bury CAD under
   "Renderings"-style subfolders.
4. Fetch *only* the files worth having: real STEP files first, then archives
   that plausibly contain STEP.
5. Unpack ZIPs and keep the STEP entries under the size cap.

Idempotent: a slug with any existing ``<slug>__*`` output is skipped, in the
top-level out_dir and in sibling/triage dirs passed via ``--also_check``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

LINK_RE = re.compile(r"https://grabcad\.com/library/([A-Za-z0-9_\-]+)")
API = "https://grabcad.com/community/api/v1/models"

STEP_EXTS = {"step", "stp", "stpz", "step.gz"}
ARCHIVE_EXTS = {"zip", "rar", "7z"}
# Archive names hinting at a neutral CAD format worth unpacking first.
STEP_HINT_RE = re.compile(r"step|stp|iges|igs|parasolid|x_t|neutral", re.I)
# WAF/CloudFront interstitials come back as HTML, never as CAD.
WAF_MARKERS = (b"<!DOCTYPE html", b"<html", b"The request could not be satisfied")


def extract_slugs(links_md: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in LINK_RE.finditer(links_md.read_text()):
        s = m.group(1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def already_have(slug: str, dirs: list[Path]) -> bool:
    for d in dirs:
        if not d.exists():
            continue
        for f in d.rglob(f"{slug}__*"):
            if f.is_file() and f.stat().st_size > 0:
                return True
    return False


def safe_stem(name: str) -> str:
    """Filesystem-safe stem; GrabCAD names contain spaces, commas, slashes."""
    stem = Path(name.replace("\\", "/")).name
    stem = Path(stem).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "part"


def get_with_backoff(req, url: str, slug: str, log: logging.Logger,
                     tries: int = 5, timeout: int = 30000):
    """GET with exponential backoff on 403.

    GrabCAD's WAF rate-limits in bursts: a run does 55-100 slugs of real work,
    then *every* remaining slug 403s within a few seconds. Without retry that
    looked like 732 permanent "misses" and took nine sequential runs to grind
    through; the slugs were never actually unavailable. Retrying with a cooldown
    turns that into one run.
    """
    delay = 5
    for attempt in range(1, tries + 1):
        try:
            r = req.get(url, headers={"Accept": "application/json"}, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] request %s: %s", slug, type(exc).__name__, exc)
            return None
        if r.status != 403:
            return r
        if attempt == tries:
            log.warning("[%s] manifest HTTP 403 after %d attempts", slug, tries)
            return r
        log.info("[%s] 403 (WAF); sleeping %ds then retrying (%d/%d)",
                 slug, delay, attempt, tries)
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return None


def list_files(req, slug: str, log: logging.Logger) -> list[dict]:
    """Return every file record for a slug, recursing through subfolders."""
    collected: list[dict] = []
    seen_ids: set = set()
    queue: list[int | None] = [None]
    seen_folders: set[int] = set()

    while queue:
        folder_id = queue.pop(0)
        url = f"{API}/{slug}/files"
        if folder_id is not None:
            url += f"?folder_id={folder_id}"
        r = get_with_backoff(req, url, slug, log)
        if r is None:
            continue
        try:
            if r.status != 200:
                log.warning("[%s] manifest HTTP %d (%s)", slug, r.status, url)
                continue
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] manifest error: %s: %s", slug, type(exc).__name__, exc)
            continue

        for f in data.get("files") or []:
            fid = f.get("id") or f.get("download_url")
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            collected.append(f)
        for fol in data.get("folders") or []:
            fid = fol.get("id")
            if fid is not None and fid not in seen_folders:
                seen_folders.add(fid)
                queue.append(fid)

    return collected


def rank_candidates(files: list[dict]) -> list[dict]:
    """Order file records by how likely they are to yield a usable STEP.

    0 = a real STEP file, 1 = archive whose name hints at STEP/neutral CAD,
    2 = any other archive. Everything else is dropped.
    """
    ranked: list[tuple[int, dict]] = []
    for f in files:
        name = f.get("name") or ""
        ext = (f.get("extension") or Path(name).suffix.lstrip(".")).lower()
        if not f.get("download_url"):
            continue
        if ext in STEP_EXTS:
            ranked.append((0, f))
        elif ext in ARCHIVE_EXTS:
            ranked.append((1 if STEP_HINT_RE.search(name) else 2, f))
    ranked.sort(key=lambda t: t[0])
    return [f for _, f in ranked]


def looks_like_html(blob: bytes) -> bool:
    head = blob[:2048]
    return any(m in head for m in WAF_MARKERS)


def resolve_download(req, url: str, log: logging.Logger, slug: str, name: str,
                     max_bytes: int | None = None) -> bytes | None:
    """Fetch a file, following GrabCAD's two-step signed-URL indirection.

    ``download_url`` does not serve the file. It returns a small JSON envelope
    ``{"url": "https://grabcad.com/cads/files/<cadid>/original.stp"}`` and the
    bytes live at that second URL. Missing this is why the first version of
    this script rejected everything as "lacks ISO-10303-21 header".
    """
    if url.startswith("/"):
        url = "https://grabcad.com" + url
    try:
        r1 = req.get(url, headers={"Accept": "application/json"}, timeout=60000)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] %s step1 %s: %s", slug, name, type(exc).__name__, exc)
        return None
    if r1.status != 200:
        log.info("[%s] %s step1 HTTP %d", slug, name, r1.status)
        return None

    body = r1.body()
    # Some entries may serve bytes directly; only treat tiny JSON as an envelope.
    inner = None
    if len(body) < 4096 and b'"url"' in body:
        try:
            inner = r1.json().get("url")
        except Exception:  # noqa: BLE001
            inner = None
    if not inner:
        return body

    # Size pre-check before pulling the payload. The size cap used to be applied
    # only *after* download, so oversized models were fetched in full and then
    # discarded: a 368 MB axle and a 282 MB engine block cost 18-28 minutes per
    # run each, and because they sat at the head of the slug list they consumed
    # whole runs' budgets. A HEAD is also the only safe guard against responses
    # Playwright cannot even hold — one 512 MB+ file raised "Cannot create a
    # string longer than 0x1fffffe8 characters".
    if max_bytes is not None:
        try:
            h = req.head(inner, timeout=60000)
            clen = int(h.headers.get("content-length") or 0)
            if clen > max_bytes:
                log.info("[%s] %s too large by HEAD (%d bytes), not downloading",
                         slug, name, clen)
                return None
        except Exception:  # noqa: BLE001
            # HEAD is advisory; if the CDN refuses it, fall through and let the
            # post-download size check catch an oversized file as before.
            pass

    try:
        r2 = req.get(inner, timeout=300000)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] %s step2 %s: %s", slug, name, type(exc).__name__, exc)
        return None
    if r2.status != 200:
        log.info("[%s] %s step2 HTTP %d", slug, name, r2.status)
        return None
    return r2.body()


def keep_zip(blob: bytes, tmp: Path, out_dir: Path, slug: str, max_bytes: int, log) -> int:
    tmp.write_bytes(blob)
    kept = 0
    try:
        with zipfile.ZipFile(tmp) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower().lstrip(".")
                if ext not in STEP_EXTS:
                    continue
                if info.file_size > max_bytes:
                    log.info("[%s] zip entry too big: %s (%d)", slug, info.filename, info.file_size)
                    continue
                target = out_dir / f"{slug}__{safe_stem(info.filename)}.step"
                n = 1
                while target.exists() and target.stat().st_size > 0:
                    target = out_dir / f"{slug}__{safe_stem(info.filename)}_{n}.step"
                    n += 1
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                kept += 1
                log.info("[%s] extracted %s (%d bytes)", slug, target.name, info.file_size)
    except zipfile.BadZipFile:
        log.info("[%s] not a valid ZIP", slug)
    finally:
        tmp.unlink(missing_ok=True)
    return kept


def fetch_slug(req, slug: str, out_dir: Path, tmp_dir: Path, max_bytes: int,
               max_tries: int, log: logging.Logger, all_step: bool = False) -> int:
    """Download STEP geometry for one slug.

    By default this stops at the first candidate that yields a file — cheapest
    way to get broad slug coverage. With ``all_step`` it keeps going through
    every candidate instead, which matters when the corpus target is large:
    GrabCAD models routinely ship several STEP variants (assembly plus
    individual bodies, or AP203/AP214 pairs), and the old archive-based
    downloader averaged ~3.1 files per successful slug precisely because it
    took everything. Duplicate geometry is fine here — preprocessing dedupes
    on face count and these are pseudo-labelled samples, not a benchmark.
    """
    files = list_files(req, slug, log)
    if not files:
        log.warning("[%s] manifest empty", slug)
        return 0

    cands = rank_candidates(files)
    if not cands:
        exts = sorted({(f.get("extension") or "?").lower() for f in files})
        log.info("[%s] no STEP/archive candidates (saw: %s)", slug, ",".join(exts))
        return 0

    def is_direct_step(rec: dict) -> bool:
        ext = (rec.get("extension") or Path(rec.get("name") or "").suffix.lstrip(".")).lower()
        return ext in STEP_EXTS

    if all_step:
        # Harvest every direct STEP file; only fall back to archives if none of
        # them produced anything. Downloading archives once we already have
        # real STEP geometry just burns bandwidth on redundant copies.
        direct = [f for f in cands if is_direct_step(f)]
        archives = [f for f in cands if not is_direct_step(f)]
        order = direct[:max_tries] if direct else archives[:max_tries]
        fallback = archives[:max_tries] if direct else []
    else:
        order = cands[:max_tries]
        fallback = []

    kept = 0
    for f in order:
        name = f["name"]
        blob = resolve_download(req, f["download_url"], log, slug, name, max_bytes)
        if blob is None:
            continue

        if looks_like_html(blob):
            log.warning("[%s] %s -> HTML/WAF page, not a file", slug, name)
            continue
        if len(blob) > max_bytes and not name.lower().endswith(tuple(ARCHIVE_EXTS)):
            log.info("[%s] %s too large (%d bytes)", slug, name, len(blob))
            continue

        ext = (f.get("extension") or Path(name).suffix.lstrip(".")).lower()
        if ext in ARCHIVE_EXTS:
            kept += keep_zip(blob, tmp_dir / f"{slug}.zip", out_dir, slug, max_bytes, log)
        else:
            if b"ISO-10303-21" not in blob[:4096]:
                log.info("[%s] %s lacks ISO-10303-21 header, skipping", slug, name)
                continue
            # Deterministic name, so a revisit pass (e.g. a second --all_step
            # run over slugs that already yielded one file) skips what it has
            # instead of writing near-duplicates with _1/_2 suffixes.
            target = out_dir / f"{slug}__{safe_stem(name)}.step"
            if target.exists() and target.stat().st_size > 0:
                log.info("[%s] already have %s", slug, target.name)
                continue
            target.write_bytes(blob)
            kept += 1
            log.info("[%s] wrote %s (%d bytes)", slug, target.name, len(blob))

        if kept and not all_step:
            break  # one good STEP source per slug is enough

    if kept == 0 and fallback:
        log.info("[%s] no direct STEP downloaded; trying %d archive(s)", slug, len(fallback))
        for f in fallback:
            name = f["name"]
            blob = resolve_download(req, f["download_url"], log, slug, name, max_bytes)
            if blob is None or looks_like_html(blob):
                continue
            kept += keep_zip(blob, tmp_dir / f"{slug}.zip", out_dir, slug, max_bytes, log)
            if kept:
                break

    return kept


def assert_signed_in(req, log: logging.Logger) -> bool:
    """Verify auth by actually downloading bytes, not by sniffing the DOM.

    The previous run's ``.avatar`` check matched GrabCAD's *anonymous* library
    page, so it reported success, then every one of 396 slugs 401'd. Only a
    real file fetch settles it — the manifest endpoint is public.
    """
    # Works whether this file is run as a script (`python .../download_grabcad_api.py`)
    # or as a module (`python -m brepformer.data.download_grabcad_api`).
    try:
        from brepformer.data.grabcad_login import api_says_logged_in
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from grabcad_login import api_says_logged_in  # type: ignore[no-redef]

    ok, detail = api_says_logged_in(req)
    if ok:
        log.info("Auth verified: %s", detail)
    else:
        log.error("NOT authenticated: %s", detail)
        log.error("Run `python brepformer/data/grabcad_login.py` and sign in first.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links_md", type=Path, default=Path("LINKS.md"))
    p.add_argument("--slugs_file", type=Path, default=None,
                   help="Newline-separated slugs; overrides --links_md")
    p.add_argument("--out_dir", type=Path, default=Path("brepformer/data/hard_samples"))
    p.add_argument("--also_check", type=Path, nargs="*", default=[
        Path("brepformer/data/hard_samples_xl_skipped")],
        help="Extra dirs consulted for idempotency (not written to)")
    p.add_argument("--tmp_dir", type=Path, default=Path("/tmp/grabcad_api"))
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--max_size_mb", type=int, default=64)
    p.add_argument("--max_tries", type=int, default=6,
                   help="Candidate files to attempt per slug before giving up")
    p.add_argument("--revisit", action="store_true",
                   help="Re-examine slugs that already have output (pairs with "
                        "--all_step to collect STEP variants missed by an earlier "
                        "first-hit-only pass). Per-file dedupe still applies.")
    p.add_argument("--all_step", action="store_true",
                   help="Keep every direct STEP file per slug, not just the first. "
                        "Raises yield from ~1 to ~3 files/slug — use when the corpus "
                        "target is large. Falls back to archives only if no direct "
                        "STEP downloaded.")
    p.add_argument("--sign_in_timeout", type=int, default=180)
    p.add_argument("--allow_unauthenticated", action="store_true",
                   help="Proceed even if the auth check fails (diagnostics only)")
    p.add_argument("--log_file", type=Path, default=Path("hard_samples_download.log"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stop_after", type=int, default=None,
                   help="Stop once this many new STEP files have been kept")
    p.add_argument("--manifest_out", type=Path, default=Path("/tmp/grabcad_api_manifest.json"))
    p.add_argument("--skip_slugs", type=Path,
                   default=Path(__file__).with_name("grabcad_skip_slugs.txt"),
                   help="File of slugs to never attempt (see that file for why). "
                        "Pass /dev/null to disable.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file, mode="a"), logging.StreamHandler()],
    )
    log = logging.getLogger("grabcad-api")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.slugs_file:
        slugs = [s for s in args.slugs_file.read_text().split() if s]
    else:
        slugs = extract_slugs(args.links_md)
    skip: set[str] = set()
    if args.skip_slugs and args.skip_slugs.exists():
        skip = {
            ln.strip() for ln in args.skip_slugs.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        }
    if skip:
        before = len(slugs)
        slugs = [s for s in slugs if s not in skip]
        log.info("skip list: dropped %d of %d slugs (%s)",
                 before - len(slugs), before, args.skip_slugs)

    if args.limit:
        slugs = slugs[: args.limit]

    check_dirs = [args.out_dir, *args.also_check]
    max_bytes = args.max_size_mb * 1024 * 1024
    log.info("API downloader: %d slugs, cap %d MB", len(slugs), args.max_size_mb)

    summary = {"kept": 0, "skipped": 0, "missed": 0, "per_slug": {}}
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            accept_downloads=True, viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        page.goto("https://grabcad.com/library", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)

        if not assert_signed_in(ctx.request, log) and not args.allow_unauthenticated:
            ctx.close()
            raise SystemExit(
                "Aborting: session is not authenticated, every download would 401.\n"
                "Fix: python brepformer/data/grabcad_login.py"
            )

        for idx, slug in enumerate(slugs, 1):
            # In --all_step revisit mode, slug-level skipping would hide the
            # extra STEP variants on slugs that already yielded one file;
            # per-file dedupe in fetch_slug handles idempotency instead.
            if not args.revisit and already_have(slug, check_dirs):
                summary["skipped"] += 1
                continue
            try:
                got = fetch_slug(ctx.request, slug, args.out_dir, args.tmp_dir,
                                 max_bytes, args.max_tries, log, all_step=args.all_step)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] unexpected: %s", slug, exc)
                got = 0
            summary["per_slug"][slug] = got
            if got:
                summary["kept"] += got
            else:
                summary["missed"] += 1

            if idx % 20 == 0:
                log.info("Progress %d/%d kept=%d skipped=%d missed=%d",
                         idx, len(slugs), summary["kept"], summary["skipped"], summary["missed"])
                args.manifest_out.write_text(json.dumps(summary, indent=2))
            if args.stop_after and summary["kept"] >= args.stop_after:
                log.info("Reached --stop_after=%d, halting.", args.stop_after)
                break

        ctx.close()

    args.manifest_out.write_text(json.dumps(summary, indent=2))
    log.info("Done. kept=%d skipped=%d missed=%d", summary["kept"], summary["skipped"], summary["missed"])


if __name__ == "__main__":
    main()
