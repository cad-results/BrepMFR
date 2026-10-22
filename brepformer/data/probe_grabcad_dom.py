"""Diagnostic: dump the download-related DOM of a few GrabCAD model pages.

Used to work out why ``download_grabcad_playwright.py``'s
``top_button_selectors`` list misses most pages. Prints every clickable
element whose text/attrs mention "download", plus the page title and
whether we look signed in.

    python brepformer/data/probe_grabcad_dom.py --slugs_file /tmp/missed_slugs.txt --n 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright

PROBE_JS = """
() => {
  const out = {signed_in_markers: [], candidates: []};
  for (const sel of ["a[href*='logout']", "[data-testid='user-menu']", ".avatar",
                     ".header-user", "img[alt*='avatar' i]", "a[href*='/dashboard']"]) {
    if (document.querySelector(sel)) out.signed_in_markers.push(sel);
  }
  const els = document.querySelectorAll('a, button, [role=button], div[onclick], span[onclick]');
  for (const el of els) {
    const txt = (el.innerText || el.textContent || '').trim().slice(0, 80);
    const attrs = {};
    for (const a of el.attributes) attrs[a.name] = String(a.value).slice(0, 160);
    const blob = (txt + ' ' + JSON.stringify(attrs)).toLowerCase();
    if (!blob.includes('download') && !blob.includes('.step') && !blob.includes('.stp')) continue;
    const r = el.getBoundingClientRect();
    out.candidates.push({
      tag: el.tagName.toLowerCase(), text: txt, attrs,
      visible: r.width > 0 && r.height > 0,
      box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
    });
  }
  return out;
}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slugs_file", type=Path, default=Path("/tmp/missed_slugs.txt"))
    p.add_argument("--slug", action="append", default=[], help="Probe a specific slug (repeatable)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--out", type=Path, default=Path("/tmp/grabcad_dom_probe.json"))
    args = p.parse_args()

    slugs = args.slug or args.slugs_file.read_text().split()[: args.n]

    results = {}
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir),
            headless=args.headless,
            accept_downloads=True,
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        for slug in slugs:
            url = f"https://grabcad.com/library/{slug}"
            entry = {"url": url}
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(3000)
                entry["title"] = page.title()
                entry.update(page.evaluate(PROBE_JS))
            except PWTimeout as exc:
                entry["error"] = f"timeout: {exc}"
            except Exception as exc:  # noqa: BLE001
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results[slug] = entry
            print(f"\n=== {slug} ===")
            print("  title:", entry.get("title"), "| error:", entry.get("error"))
            print("  signed_in:", entry.get("signed_in_markers"))
            for c in entry.get("candidates", []):
                print(f"  [{c['visible']}] <{c['tag']}> {c['text']!r} {c['attrs']}")
        ctx.close()

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
