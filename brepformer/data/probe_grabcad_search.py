"""Diagnostic: find a GrabCAD search/listing endpoint that yields model slugs.

To reach ~1000 *distinct* parts we need several hundred more slugs than
``LINKS.md`` has. Hand-curating those is slow and biases toward whatever a
search UI surfaces first; if the community API exposes a paginated
search/listing endpoint we can harvest slugs programmatically and filter by
the metadata that actually predicts a useful training sample.

We already know ``/community/api/v1/models/<slug>`` and
``/community/api/v1/models/<slug>/files`` are public via ``context.request``.
This probes plausible siblings for listing/search, plus the filter params
(software=STEP, category, sort, page) that would let us bias toward
single-body machined parts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://grabcad.com/community/api/v1"

CANDIDATES = [
    f"{BASE}/models?page=1&per_page=24",
    f"{BASE}/models?query=bracket&page=1",
    f"{BASE}/models/search?query=bracket&page=1",
    f"{BASE}/search?query=bracket&page=1",
    f"{BASE}/search/models?query=bracket&page=1",
    f"{BASE}/library?page=1",
    "https://grabcad.com/library?softwares=step&page=1",
    "https://grabcad.com/community/api/v1/models?softwares=step&page=1&per_page=24",
]


def summarize(body: str) -> str:
    """Report the shape of a JSON response and any slug-ish keys in it."""
    try:
        data = json.loads(body)
    except Exception:
        return f"(non-JSON) {body[:200]!r}"

    def find_slugs(o, depth=0, acc=None):
        if acc is None:
            acc = []
        if depth > 5 or len(acc) > 5:
            return acc
        if isinstance(o, dict):
            for k in ("cached_slug", "slug", "url", "id"):
                if k in o and isinstance(o[k], str):
                    acc.append(f"{k}={o[k][:60]}")
                    break
            for v in o.values():
                find_slugs(v, depth + 1, acc)
        elif isinstance(o, list):
            for v in o[:5]:
                find_slugs(v, depth + 1, acc)
        return acc

    top = list(data.keys())[:12] if isinstance(data, dict) else f"list[{len(data)}]"
    return f"keys={top} slugs={find_slugs(data)[:5]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--out", type=Path, default=Path("/tmp/grabcad_search_probe.json"))
    args = p.parse_args()

    results = {}
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            accept_downloads=True, viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        page.goto("https://grabcad.com/library", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)

        for url in CANDIDATES:
            try:
                r = ctx.request.get(url, headers={"Accept": "application/json"}, timeout=30000)
                body = r.text()
                line = f"{r.status} ct={r.headers.get('content-type','')[:40]} len={len(body)}"
                detail = summarize(body) if r.status == 200 else body[:150]
                print(f"\n### {url}\n  {line}\n  {detail}")
                results[url] = {"status": r.status, "len": len(body), "sample": body[:3000]}
            except Exception as exc:  # noqa: BLE001
                print(f"\n### {url}\n  ERROR {type(exc).__name__}: {exc}")
                results[url] = {"error": str(exc)}

        # Also capture what the real library page's own XHRs hit — the most
        # reliable way to learn the endpoint the UI actually uses.
        seen: list[str] = []
        page.on("request", lambda req: seen.append(req.url)
                if "/api/" in req.url or "search" in req.url else None)
        page.goto("https://grabcad.com/library?softwares=step", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(6000)
        try:
            page.mouse.wheel(0, 4000)
            page.wait_for_timeout(4000)
        except Exception:  # noqa: BLE001
            pass
        print("\n### XHRs observed on the library page:")
        for u in dict.fromkeys(seen):
            print("   ", u[:180])
        results["_observed_xhr"] = list(dict.fromkeys(seen))

        ctx.close()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
