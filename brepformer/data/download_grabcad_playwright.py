"""Playwright-based GrabCAD downloader for STEP training data.

Runs a real Chromium under a persistent user-data directory so the AWS WAF
challenge is solved by the actual browser and the session survives runs.

Workflow:
1. Launch a **headed** Chromium (browser window visible).
2. First run: the user manually signs into grabcad.com. State is persisted
   to ``--user_data_dir`` (default: ``~/.grabcad_playwright``). Subsequent
   runs reuse that state.
3. For each slug parsed from ``LINKS.md``:
    a. Navigate to ``https://grabcad.com/library/<slug>``.
    b. Locate a STEP file entry (or a "Download" button that yields a ZIP
       containing the STEP).
    c. Use ``page.expect_download()`` to trigger + save the file.
    d. If a ZIP arrives, unpack it and keep only ``.step`` / ``.stp`` entries
       under ``<max_size_mb>``.
4. Idempotent: slugs whose output already exists on disk are skipped.

The downloader logs per-slug outcomes to ``hard_samples_download.log``.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import zipfile
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PWTimeout, sync_playwright

LINK_RE = re.compile(r"https://grabcad\.com/library/([A-Za-z0-9_\-]+)")
STEP_EXTS = (".step", ".stp")


def extract_slugs(links_md: Path) -> list[str]:
    seen: set[str] = set()
    slugs: list[str] = []
    for m in LINK_RE.finditer(links_md.read_text()):
        s = m.group(1)
        if s in seen:
            continue
        seen.add(s)
        slugs.append(s)
    return slugs


def wait_for_signed_in(page: Page, timeout_s: int = 300) -> None:
    """Block until we see an authenticated marker on GrabCAD."""
    log = logging.getLogger("grabcad")
    log.info(
        "Waiting up to %ds for you to sign in. Complete any WAF challenge "
        "and log in via the browser window.",
        timeout_s,
    )
    page.goto("https://grabcad.com/", wait_until="domcontentloaded")
    # `.avatar`, `[data-testid="user-menu"]`, or a "logout" anchor should
    # appear after login. Poll for any of them.
    end_selector = (
        "a[href*='logout'], "
        "[data-testid='user-menu'], "
        ".avatar, "
        ".header-user, "
        "img[alt*='avatar' i]"
    )
    try:
        page.wait_for_selector(end_selector, timeout=timeout_s * 1000)
        log.info("Detected signed-in state.")
    except PWTimeout:
        log.warning(
            "No sign-in marker detected within %ds — continuing anyway. "
            "If downloads 401/403, run again and log in first.",
            timeout_s,
        )


def is_step_name(name: str) -> bool:
    return name.lower().endswith(STEP_EXTS)


def handle_zip(zip_path: Path, out_dir: Path, slug: str, max_bytes: int, log: logging.Logger) -> int:
    """Extract every STEP entry from a ZIP into ``out_dir``. Returns count."""
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not is_step_name(info.filename):
                    continue
                if info.file_size > max_bytes:
                    log.info(
                        "[%s] skipping %s (%d bytes > cap)",
                        slug, info.filename, info.file_size,
                    )
                    continue
                stem = Path(info.filename).stem
                ext = Path(info.filename).suffix.lower()
                if ext not in STEP_EXTS:
                    ext = ".step"
                target = out_dir / f"{slug}__{stem}{ext}"
                if target.exists() and target.stat().st_size > 0:
                    extracted += 1
                    continue
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
                log.info("[%s] wrote %s (%d bytes)", slug, target.name, info.file_size)
    except zipfile.BadZipFile:
        log.warning("[%s] downloaded file is not a valid ZIP: %s", slug, zip_path)
    return extracted


def download_slug(
    page: Page,
    slug: str,
    out_dir: Path,
    tmp_dir: Path,
    max_bytes: int,
    log: logging.Logger,
) -> int:
    """Navigate to the model page and trigger the download. Returns files kept."""
    # Idempotency: if any file starting with `<slug>__` exists, skip.
    if any(out_dir.glob(f"{slug}__*")):
        return -1  # sentinel: already done

    url = f"https://grabcad.com/library/{slug}"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except PWTimeout:
        log.warning("[%s] page load timed out", slug)
        return 0

    # 1) First look for direct file-link anchors (some pages expose STEP
    #    downloads via a plain <a href=".../download"> without a modal).
    # 2) Otherwise fall back to the top-level Download button which opens
    #    a modal listing files — then click the first STEP entry inside.
    # Preference order — the plain "Download" anchor is common in the
    # sidebar (goes nowhere), so try the specific button labels first.
    top_button_selectors = [
        "button:has-text('Download files')",
        "button:has-text('Download CAD')",
        "a:has-text('Download files')",
        "a:has-text('Download CAD')",
        ".model-download-button",
        "[data-testid*='download' i] button",
        "button:has-text('Download')",
        "a:has-text('Download')",
    ]
    click_target = None
    for sel in top_button_selectors:
        try:
            cand = page.locator(sel).first
            cand.wait_for(state="visible", timeout=6000)
            click_target = cand
            break
        except PWTimeout:
            continue
    if click_target is None:
        log.warning("[%s] no download button found", slug)
        return 0

    dl = None
    # Attempt 1: click the top button, hope for a direct download.
    try:
        with page.expect_download(timeout=15000) as dl_info:
            click_target.click()
        dl = dl_info.value
    except PWTimeout:
        pass

    # Attempt 2: if a modal opened, find a STEP row inside and click it.
    if dl is None:
        step_row = page.locator(
            "a[href*='.step'], a[href*='.stp'], "
            "a[href*='/download']:has-text('.step'), "
            "a[href*='/download']:has-text('.stp'), "
            "[role='dialog'] a:has-text('.step'), "
            "[role='dialog'] a:has-text('.stp'), "
            "[role='dialog'] li:has-text('.step') a, "
            "[role='dialog'] li:has-text('.stp') a"
        ).first
        try:
            step_row.wait_for(state="visible", timeout=15000)
            with page.expect_download(timeout=60000) as dl_info:
                step_row.click()
            dl = dl_info.value
        except PWTimeout:
            log.warning("[%s] no download fired (modal fallback also missed)", slug)
            return 0

    suggested = dl.suggested_filename or f"{slug}.bin"
    tmp_path = tmp_dir / f"{slug}__{suggested}"
    dl.save_as(tmp_path)

    kept = 0
    if suggested.lower().endswith(".zip"):
        kept = handle_zip(tmp_path, out_dir, slug, max_bytes, log)
        tmp_path.unlink(missing_ok=True)
    elif is_step_name(suggested):
        size = tmp_path.stat().st_size
        if size > max_bytes:
            log.info("[%s] step file too large: %d bytes", slug, size)
            tmp_path.unlink(missing_ok=True)
        else:
            ext = Path(suggested).suffix.lower()
            target = out_dir / f"{slug}__{Path(suggested).stem}{ext}"
            tmp_path.replace(target)
            log.info("[%s] wrote %s (%d bytes)", slug, target.name, size)
            kept = 1
    else:
        log.info("[%s] downloaded non-ZIP non-STEP file: %s", slug, suggested)
        tmp_path.unlink(missing_ok=True)

    return kept


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links_md", type=Path, default=Path("LINKS.md"))
    p.add_argument("--out_dir", type=Path, default=Path("brepformer/data/hard_samples"))
    p.add_argument("--tmp_dir", type=Path, default=Path("/tmp/grabcad_downloads"))
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--max_size_mb", type=int, default=64)
    p.add_argument("--headless", action="store_true", help="Skip the headed sign-in step (reuses saved cookies)")
    p.add_argument("--sign_in_timeout", type=int, default=300)
    p.add_argument("--log_file", type=Path, default=Path("hard_samples_download.log"))
    p.add_argument("--limit", type=int, default=None, help="Only process the first N slugs (smoke test)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file, mode="a"), logging.StreamHandler()],
    )
    log = logging.getLogger("grabcad")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.user_data_dir.mkdir(parents=True, exist_ok=True)

    slugs = extract_slugs(args.links_md)
    if args.limit:
        slugs = slugs[: args.limit]
    log.info("Parsed %d slugs from %s", len(slugs), args.links_md)

    max_bytes = args.max_size_mb * 1024 * 1024

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir),
            headless=args.headless,
            accept_downloads=True,
            viewport={"width": 1400, "height": 900},
        )
        page = context.new_page()

        if not args.headless:
            wait_for_signed_in(page, timeout_s=args.sign_in_timeout)

        total_kept = 0
        total_skipped = 0
        total_missed = 0
        for idx, slug in enumerate(slugs, 1):
            try:
                got = download_slug(page, slug, args.out_dir, args.tmp_dir, max_bytes, log)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] unexpected failure: %s", slug, exc)
                got = 0
            if got < 0:
                total_skipped += 1
            elif got == 0:
                total_missed += 1
            else:
                total_kept += got
            if idx % 25 == 0:
                log.info(
                    "Progress %d/%d — kept=%d skipped=%d missed=%d",
                    idx, len(slugs), total_kept, total_skipped, total_missed,
                )

        log.info(
            "Done. kept=%d skipped_existing=%d missed=%d slugs=%d",
            total_kept, total_skipped, total_missed, len(slugs),
        )
        context.close()


if __name__ == "__main__":
    main()
