"""Diagnostic: pull GrabCAD's per-model file list out of the Angular scope.

The model page is an AngularJS app whose scope holds a ``files`` array where
each entry has ``name`` + ``download_url``. Reading that is far more robust
than clicking the "Download files" button (which builds a server-side ZIP of
*every* file and often never fires a browser download event).

    python brepformer/data/probe_grabcad_files.py --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

# Walk every Angular scope reachable from the DOM and collect anything that
# looks like a file record (has a name and a download_url).
SCOPE_JS = """
() => {
  const found = [];
  const seen = new Set();
  const push = (o) => {
    if (!o || typeof o !== 'object') return;
    if (typeof o.name === 'string' && typeof o.download_url === 'string') {
      const k = o.name + '|' + o.download_url;
      if (!seen.has(k)) { seen.add(k); found.push({name: o.name, download_url: o.download_url, size: o.size ?? null}); }
    }
  };
  const walk = (o, depth) => {
    if (depth > 6 || !o || typeof o !== 'object') return;
    push(o);
    for (const k of Object.keys(o)) {
      if (k.startsWith('$$') || k === '$parent' || k === '$root') continue;
      const v = o[k];
      if (Array.isArray(v)) { for (const it of v) walk(it, depth + 1); }
      else if (v && typeof v === 'object') walk(v, depth + 1);
    }
  };

  const out = {mode: null, files: [], archive_url: null};
  if (window.angular) {
    out.mode = 'angular';
    const roots = new Set();
    document.querySelectorAll('*').forEach(el => {
      try {
        const s = window.angular.element(el).scope();
        if (s) roots.add(s);
      } catch (e) {}
    });
    for (const s of roots) {
      walk(s, 0);
      if (s.model && s.model.archive_url) out.archive_url = s.model.archive_url;
    }
  }
  // Fallback: scrape ng-click handlers verbatim so we can at least see the shape.
  if (!out.files.length) {
    document.querySelectorAll('[ng-click*="downloadFile"]').forEach(el => {
      out.files.push({name: null, download_url: null, ng_click: el.getAttribute('ng-click')});
    });
  } else {
    out.files = found;
  }
  out.files = found.length ? found : out.files;
  return out;
}
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slugs_file", type=Path, default=Path("/tmp/missed_slugs.txt"))
    p.add_argument("--slug", action="append", default=[])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--out", type=Path, default=Path("/tmp/grabcad_files_probe.json"))
    args = p.parse_args()

    slugs = args.slug or args.slugs_file.read_text().split()[: args.n]
    results = {}
    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            accept_downloads=True, viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        for slug in slugs:
            try:
                page.goto(f"https://grabcad.com/library/{slug}", wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(3500)
                info = page.evaluate(SCOPE_JS)
            except Exception as exc:  # noqa: BLE001
                info = {"error": f"{type(exc).__name__}: {exc}"}
            results[slug] = info
            print(f"\n=== {slug} === mode={info.get('mode')} archive={str(info.get('archive_url'))[:80]}")
            for f in (info.get("files") or [])[:15]:
                print("   ", json.dumps(f)[:220])
            if info.get("error"):
                print("    ERROR:", info["error"])
        ctx.close()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
