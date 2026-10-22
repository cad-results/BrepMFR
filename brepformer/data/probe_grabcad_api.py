"""Diagnostic: can we hit GrabCAD's community REST API from the browser context?

The model page's file entries point at
``/community/api/v1/models/<slug>/files/download?cadid=...``, which implies a
sibling list endpoint. If a JSON manifest is reachable without rendering the
whole AngularJS page, Phase 2 gets ~10x faster and stops depending on button
selectors entirely.

Tries several candidate endpoints two ways:
  * ``context.request`` (APIRequestContext — shares cookies, no JS)
  * in-page ``fetch()`` from an already-loaded grabcad.com origin

    python brepformer/data/probe_grabcad_api.py --slug grass-3803-hinge-slash-120-degree-opening
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

CANDIDATES = [
    "https://grabcad.com/community/api/v1/models/{slug}",
    "https://grabcad.com/community/api/v1/models/{slug}/files",
    "https://grabcad.com/community/api/v1/models/{slug}?include=files",
    "https://grabcad.com/library/{slug}.json",
]

FETCH_JS = """
async (url) => {
  try {
    const r = await fetch(url, {credentials: 'include', headers: {'Accept': 'application/json'}});
    const t = await r.text();
    return {status: r.status, ct: r.headers.get('content-type'), body: t.slice(0, 1200), len: t.length};
  } catch (e) { return {error: String(e)}; }
}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default="grass-3803-hinge-slash-120-degree-opening")
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    args = p.parse_args()

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            accept_downloads=True, viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        # Land on the origin first so in-page fetch() is same-origin and the
        # WAF cookie is attached.
        page.goto(f"https://grabcad.com/library/{args.slug}", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)

        for tmpl in CANDIDATES:
            url = tmpl.format(slug=args.slug)
            print(f"\n### {url}")

            try:
                r = ctx.request.get(url, headers={"Accept": "application/json"})
                body = r.text()
                print(f"  [context.request] {r.status} ct={r.headers.get('content-type')} len={len(body)}")
                print("   ", body[:400].replace("\n", " "))
            except Exception as exc:  # noqa: BLE001
                print(f"  [context.request] ERROR {type(exc).__name__}: {exc}")

            res = page.evaluate(FETCH_JS, url)
            print(f"  [in-page fetch] {json.dumps(res)[:600]}")

        ctx.close()


if __name__ == "__main__":
    main()
