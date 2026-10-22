"""Open a headed Chromium on the shared GrabCAD profile and wait for real login.

Why this is a separate step
--------------------------
The Phase 2 run that "missed 396 slugs" was not a selector problem. The old
``wait_for_signed_in`` polled for ``.avatar`` and treated a match as success,
but GrabCAD's anonymous library page also satisfied that selector, so the run
proceeded unauthenticated and every file download came back
``401 {"error":"You need to log in or sign up before continuing."}``.

So auth gets its own script with an *authoritative* check: browsing state is
irrelevant, what matters is whether a real file ``download_url`` returns
bytes. We assert that against a known-good public model before declaring
success.

Usage — sign in interactively (the window stays open until you're done):

    python brepformer/data/grabcad_login.py

Then confirm any time without touching the UI:

    python brepformer/data/grabcad_login.py --check_only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "https://grabcad.com/community/api/v1/models"
# Any published model with a real CAD file works as the auth canary.
CANARY_SLUG = "cnc-laser-cutter-1"


def dom_says_logged_in(page) -> bool:
    """Heuristic, DOM-level check: a logout affordance and no login prompt."""
    try:
        logged_out = page.locator("a:has-text('Log in'), a[href*='login']").count()
        logged_in = page.locator("a[href*='logout'], [data-testid='user-menu']").count()
        return logged_in > 0 and logged_out == 0
    except Exception:  # noqa: BLE001
        return False


def api_says_logged_in(req, slug: str = CANARY_SLUG) -> tuple[bool, str]:
    """Authoritative check: does a real file download return CAD bytes?

    The manifest endpoint is public, so a 200 there proves nothing. Only the
    per-file ``download_url`` distinguishes an authenticated session.
    """
    try:
        man = req.get(f"{API}/{slug}/files", headers={"Accept": "application/json"}, timeout=30000)
        if man.status != 200:
            return False, f"manifest HTTP {man.status} (WAF token may be stale)"
        files = man.json().get("files") or []
        target = next((f for f in files
                       if (f.get("extension") or "").lower() in ("step", "stp")), None)
        if target is None:
            return False, "canary model exposes no STEP file; pick another --canary"

        url = target["download_url"]
        if url.startswith("/"):
            url = "https://grabcad.com" + url

        # Step 1 returns a JSON envelope, not the file: {"url": ".../original.stp"}
        r1 = req.get(url, headers={"Accept": "application/json"}, timeout=60000)
        body = r1.body()
        if r1.status in (401, 403) or b"log in or sign up" in body:
            return False, f"download step1 HTTP {r1.status}: {body[:120]!r}"
        if r1.status != 200:
            return False, f"download step1 HTTP {r1.status}"
        try:
            inner = r1.json().get("url")
        except Exception:  # noqa: BLE001
            inner = None
        if not inner:
            return False, f"step1 gave no signed url: {body[:120]!r}"

        # Step 2 is the real test. Range-limit it — a few KB is enough to tell
        # a STEP header from a login-wall response, and canary files can be 50MB+.
        r2 = req.get(inner, headers={"Range": "bytes=0-2047"}, timeout=60000)
        b2 = r2.body()
        if r2.status in (401, 403) or b"log in or sign up" in b2:
            return False, f"download step2 HTTP {r2.status}: {b2[:120]!r}"
        if r2.status not in (200, 206):
            return False, f"download step2 HTTP {r2.status}"
        if b"ISO-10303-21" not in b2[:4096]:
            return False, f"step2 bytes are not STEP: {b2[:80]!r}"
        return True, f"step2 HTTP {r2.status}, got STEP bytes ({len(b2)} sampled)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--canary", default=CANARY_SLUG)
    p.add_argument("--timeout", type=int, default=900, help="Seconds to wait for you to sign in")
    p.add_argument("--poll", type=int, default=10)
    p.add_argument("--check_only", action="store_true",
                   help="Report auth state and exit; do not wait for login")
    args = p.parse_args()

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir),
            headless=False, accept_downloads=True,
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://grabcad.com/login", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)

        ok, detail = api_says_logged_in(ctx.request, args.canary)
        print(f"[auth] initial: {'AUTHENTICATED' if ok else 'NOT authenticated'} — {detail}")
        print(f"[auth] dom heuristic: logged_in={dom_says_logged_in(page)}")

        if ok:
            print("[auth] Session is already good; nothing to do.")
            ctx.close()
            return 0
        if args.check_only:
            ctx.close()
            return 1

        print()
        print("=" * 72)
        print(" A Chromium window is open on grabcad.com/login.")
        print(" Please sign in there (complete any SSO / WAF challenge).")
        print(f" I'll re-check every {args.poll}s for up to {args.timeout}s and")
        print(" exit as soon as a real file download succeeds.")
        print(" Leave the window open — closing it aborts the wait.")
        print("=" * 72)
        sys.stdout.flush()

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            time.sleep(args.poll)
            if page.is_closed() and not ctx.pages:
                print("[auth] browser was closed; aborting.")
                break
            ok, detail = api_says_logged_in(ctx.request, args.canary)
            remaining = int(deadline - time.time())
            print(f"[auth] {'OK' if ok else 'waiting'} ({remaining}s left) — {detail}")
            sys.stdout.flush()
            if ok:
                break

        if ok:
            print("[auth] AUTHENTICATED — session persisted to", args.user_data_dir)
            print("[auth] Safe to start the bulk download now.")
        else:
            print("[auth] Still not authenticated. Re-run this script and sign in.")
        ctx.close()
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
