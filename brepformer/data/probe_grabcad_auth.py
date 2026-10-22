"""Diagnostic: what makes GrabCAD's file-download endpoint return 200 instead of 401?

The ``/community/api/v1/models/<slug>/files`` manifest returns 200 through
``context.request``, but each entry's ``download_url`` returns 401 the same
way. Something the AngularJS app sends is missing. This probes, for one real
STEP file, the axes most likely to matter:

  * transport: ``context.request`` vs in-page ``fetch()`` vs a real
    navigation/anchor click captured by ``expect_download``
  * URL form: the ``/community/api/v1/...`` form vs the ``/library/<slug>/download``
    form the page's own rows use
  * headers: Referer, X-Requested-With, Accept

Whichever combination returns CAD bytes is the one the downloader should use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "https://grabcad.com/community/api/v1/models"

FETCH_JS = """
async (url) => {
  try {
    const r = await fetch(url, {credentials: 'include'});
    const b = await r.arrayBuffer();
    const head = new TextDecoder().decode(new Uint8Array(b).slice(0, 120));
    return {status: r.status, ct: r.headers.get('content-type'), len: b.byteLength, head};
  } catch (e) { return {error: String(e)}; }
}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", default="cnc-laser-cutter-1")
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    args = p.parse_args()
    slug = args.slug

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            accept_downloads=True, viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        page_url = f"https://grabcad.com/library/{slug}"
        page.goto(page_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        man = ctx.request.get(f"{API}/{slug}/files", headers={"Accept": "application/json"})
        files = man.json().get("files") or []
        step = next((f for f in files
                     if (f.get("extension") or "").lower() in ("step", "stp")), None)
        if step is None:
            print("no STEP entry in manifest; files:", [f.get("name") for f in files])
            ctx.close()
            return
        print("target file:", json.dumps(step)[:400])

        api_url = step["download_url"]
        if api_url.startswith("/"):
            api_url = "https://grabcad.com" + api_url
        cadid = api_url.split("cadid=")[-1]
        lib_url = f"https://grabcad.com/library/{slug}/download?cadid={cadid}"

        header_sets = {
            "bare": {},
            "referer": {"Referer": page_url},
            "referer+xhr": {"Referer": page_url, "X-Requested-With": "XMLHttpRequest"},
            "referer+accept": {"Referer": page_url, "Accept": "*/*",
                               "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"},
        }
        for label, url in (("api", api_url), ("library", lib_url)):
            for hlabel, headers in header_sets.items():
                try:
                    r = ctx.request.get(url, headers=headers, timeout=60000)
                    body = r.body()
                    print(f"  [ctx.request/{label}/{hlabel}] {r.status} len={len(body)} "
                          f"ct={r.headers.get('content-type')} head={body[:60]!r}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [ctx.request/{label}/{hlabel}] ERROR {type(exc).__name__}: {exc}")
            res = page.evaluate(FETCH_JS, url)
            print(f"  [in-page fetch/{label}] {json.dumps(res)[:300]}")

        # Transport of last resort: let the browser itself do the download.
        for label, url in (("api", api_url), ("library", lib_url)):
            try:
                with page.expect_download(timeout=60000) as di:
                    page.evaluate(
                        "(u) => { const a = document.createElement('a');"
                        " a.href = u; a.download = ''; document.body.appendChild(a); a.click(); }",
                        url,
                    )
                d = di.value
                tmp = Path("/tmp") / f"probe_{label}_{d.suggested_filename}"
                d.save_as(tmp)
                print(f"  [anchor-click/{label}] OK {d.suggested_filename} "
                      f"{tmp.stat().st_size} bytes head={tmp.read_bytes()[:60]!r}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [anchor-click/{label}] {type(exc).__name__}: {str(exc)[:200]}")

        ctx.close()


if __name__ == "__main__":
    main()
