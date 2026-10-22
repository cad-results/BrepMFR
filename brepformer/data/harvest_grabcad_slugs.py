"""Harvest STEP-bearing GrabCAD slugs to grow the corpus toward 1000 parts.

Why this exists
---------------
``LINKS.md`` was hand-curated to 476 slugs. Of those, ~130-140 are permanently
unusable (no STEP published, or a single part exceeding the size cap), so the
list tops out well short of 1000 *distinct* parts. Hand-curating another 500
links is slow and biases the corpus toward whatever the search UI happens to
rank first.

``probe_grabcad_search.py`` established that the community API exposes a
paginated listing endpoint::

    /community/api/v1/models?query=<q>&page=<n>

Each record carries ``cached_slug`` plus a ``softwares`` list, e.g.
``[{"name": "STEP / IGES"}]``. Three properties of the endpoint drive this
script's design:

* **``per_page`` is ignored.** Asking for 50 returns 24. Pagination is fixed.
* **Pagination hard-caps at 100 pages.** Page 100, 101 and 150 all return the
  same records, and ``total_entries`` reports exactly 2400 (= 100 x 24) for
  every query — so that figure is the *cap*, not a count of matches. A single
  query therefore cannot yield more than ~2400 slugs, which is why we sweep a
  broad vocabulary of queries rather than paginating one deeply.
* **Server-side format filtering does not work.** ``softwares=step``,
  ``software=step`` and ``filter[softwares]=step`` all return unfiltered
  results with an unchanged total. So we filter *client-side* on the
  ``softwares`` list that the response already gives us.

That client-side filter is the point of the whole script. Roughly 37% of
listed models publish STEP/IGES; the rest are SOLIDWORKS-only, STL-only or
render-only. Filtering before download is what turns "no STEP/archive
candidates" — 49 wasted slugs in the last pass — into zero wasted requests.

Quality, not count
------------------
The corpus target is distinct *machined parts*, so this script biases toward
them rather than maximising slug count:

* the query vocabulary is machining-feature and mechanical-component
  vocabulary, aligned with what MFTRCAD actually labels;
* ``STEP / IGES`` is *necessary but not sufficient* — that one GrabCAD tag
  covers both formats, so a model tagged with it may publish IGES only. The
  downloader still has to check, and geometric curation still has the last
  word. This filter removes certain losses, it does not certify a win;
* mesh-only formats (STL/OBJ/3MF) are rejected outright: a tessellated mesh
  has no B-rep faces to label, so it cannot become a training sample;
* obvious assemblies are dropped by name, since MFTRCAD is single-body
  machined parts and the teacher's pseudo-labels are least reliable on
  multi-body input.

Output is a plain newline-separated slug list for ``download_grabcad_api.py
--slugs_file``, and optionally an appended section in ``LINKS.md``.

    # discover, write /tmp/new_slugs.txt, don't touch LINKS.md
    python brepformer/data/harvest_grabcad_slugs.py --target 800

    # also append to LINKS.md
    python brepformer/data/harvest_grabcad_slugs.py --target 800 --append_links
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

API = "https://grabcad.com/community/api/v1/models"
LINK_RE = re.compile(r"https://grabcad\.com/library/([A-Za-z0-9_\-]+)")

#: A model is worth downloading only if one of these appears in its softwares
#: list. "STEP / IGES" is GrabCAD's single tag for both formats, so it is a
#: necessary-not-sufficient signal; Parasolid/x_t archives frequently carry a
#: STEP sibling too.
NEUTRAL_SW_RE = re.compile(r"step|iges|parasolid|x_t|acis|sat\b", re.I)

#: Mesh-only publications can never yield B-rep faces, so they are not merely
#: unhelpful — they are unusable. Rejected even if something else matches.
MESH_ONLY_RE = re.compile(r"^\s*(stl|obj|3mf|ply|blender|keyshot|rendering)\s*$", re.I)

#: Multi-body assemblies are out of MFTRCAD's single-part domain and are where
#: the teacher model's confidence is measurably worst.
ASSEMBLY_NAME_RE = re.compile(
    r"assembl|assy|complete\s+set|full\s+machine|engine\b|car\b|robot\b|"
    r"gearbox|drivetrain|chassis|airplane|aircraft|motorcycle|bicycle",
    re.I,
)

#: Machining vocabulary. Deliberately weighted toward prismatic/turned parts
#: with the feature classes MFTRCAD labels (holes, slots, pockets, chamfers,
#: fillets, bosses, threads) rather than toward popular consumer models.
QUERIES = [
    # prismatic machined bodies
    "bracket", "flange", "housing", "manifold", "adapter plate", "base plate",
    "mounting plate", "end cap", "cover plate", "spacer", "block", "clamp",
    "bearing housing", "motor mount", "gearbox housing", "valve body",
    "pump housing", "cylinder head", "machined block", "fixture plate",
    "jig plate", "tooling plate", "heat sink", "enclosure machined",
    # turned / axisymmetric
    "shaft", "bushing", "bush", "sleeve", "collar", "hub", "pulley", "sprocket",
    "coupling", "spindle", "axle", "pin dowel", "stud", "nozzle", "piston",
    "cylinder liner", "spacer ring", "retaining ring", "threaded insert",
    # feature-led queries (bias toward parts that actually show MFTRCAD features)
    "counterbore", "countersink", "through hole plate", "keyway shaft",
    "t slot", "dovetail", "chamfered", "filleted", "tapped hole",
    "slotted bracket", "pocket milled", "boss machined", "rib machined",
    "knurled", "splined shaft", "o-ring groove", "snap ring groove",
    # standard mechanical components
    "gear blank", "worm gear", "bevel gear", "rack gear", "cam", "lever",
    "crank", "linkage", "yoke", "clevis", "trunnion", "gusset", "standoff",
    "hinge", "latch", "bearing block", "pillow block", "rod end", "eye bolt",
    "hex nut", "washer machined", "set screw", "grub screw", "shoulder bolt",
    # process / material framing
    "cnc part", "milled part", "turned part", "machined component",
    "sheet metal bracket", "weldment machined", "die casting", "aluminium part",
    "steel part", "mechanical part", "spare part", "machine element",
]


def load_existing(links_md: Path) -> set[str]:
    if not links_md.exists():
        return set()
    return set(LINK_RE.findall(links_md.read_text()))


def software_names(rec: dict) -> list[str]:
    out: list[str] = []
    for s in rec.get("softwares") or []:
        name = s.get("name") if isinstance(s, dict) else s
        if isinstance(name, str):
            out.append(name)
    return out


def is_wanted(rec: dict) -> tuple[bool, str]:
    """Decide whether a listing record is worth a download attempt."""
    slug = rec.get("cached_slug")
    if not slug:
        return False, "no_slug"
    if rec.get("is_hidden") or rec.get("is_tainted"):
        return False, "hidden_or_tainted"

    sw = software_names(rec)
    if not sw:
        return False, "no_software_listed"
    if all(MESH_ONLY_RE.match(s) for s in sw):
        return False, "mesh_or_render_only"
    if not any(NEUTRAL_SW_RE.search(s) for s in sw):
        return False, "no_neutral_format"

    if ASSEMBLY_NAME_RE.search(rec.get("name") or ""):
        return False, "assembly_by_name"
    return True, "ok"


def harvest(req, queries: list[str], existing: set[str], target: int,
            max_pages: int, sleep_ms: int, verbose: bool,
            max_per_query: int) -> tuple[dict[str, dict], collections.Counter]:
    """Sweep the query vocabulary, capping each query's contribution.

    The cap is not politeness, it is the whole point. Without it the first
    query saturates the target: an uncapped run for 900 slugs returned 765
    from ``bracket`` alone and never reached query 3 of 96. That reproduces the
    exact concentration problem ``curate_corpus.py`` exists to undo — a corpus
    of one part type teaches one part type — except at harvest time, where it
    is far cheaper to prevent than to detect later.
    """
    found: dict[str, dict] = {}
    reasons: collections.Counter = collections.Counter()
    consecutive_403 = 0
    per_query: collections.Counter = collections.Counter()

    for qi, q in enumerate(queries, 1):
        if len(found) >= target:
            break
        prev_first: str | None = None
        for page in range(1, max_pages + 1):
            if len(found) >= target or per_query[q] >= max_per_query:
                break
            url = f"{API}?query={q.replace(' ', '%20')}&page={page}"
            try:
                r = req.get(url, headers={"Accept": "application/json"}, timeout=30000)
            except Exception as exc:  # noqa: BLE001
                reasons[f"request_error:{type(exc).__name__}"] += 1
                break
            if r.status == 403:
                # The WAF rate-limits in bursts; back off rather than burning
                # the rest of the query list on guaranteed failures.
                consecutive_403 += 1
                reasons["http_403"] += 1
                back = min(60, 5 * consecutive_403)
                if verbose:
                    print(f"  [{q}] 403 — backing off {back}s")
                time.sleep(back)
                if consecutive_403 >= 8:
                    print("  too many consecutive 403s; stopping this sweep")
                    return found, reasons
                continue
            if r.status != 200:
                reasons[f"http_{r.status}"] += 1
                break
            consecutive_403 = 0
            try:
                models = r.json().get("models") or []
            except Exception:  # noqa: BLE001
                reasons["bad_json"] += 1
                break
            if not models:
                break

            # Pagination caps at 100 pages by returning the same page forever;
            # detect it by a repeated leading slug instead of trusting the cap.
            first = models[0].get("cached_slug")
            if first is not None and first == prev_first:
                reasons["page_repeat_stop"] += 1
                break
            prev_first = first

            for rec in models:
                if per_query[q] >= max_per_query:
                    break
                slug = rec.get("cached_slug")
                ok, why = is_wanted(rec)
                if not ok:
                    reasons[why] += 1
                    continue
                if slug in existing:
                    reasons["already_in_links"] += 1
                    continue
                if slug in found:
                    reasons["dup_across_queries"] += 1
                    continue
                found[slug] = {
                    "slug": slug,
                    "name": rec.get("name"),
                    "softwares": software_names(rec),
                    "downloads": rec.get("downloads_count"),
                    "query": q,
                }
                per_query[q] += 1
            if sleep_ms:
                time.sleep(sleep_ms / 1000)

        if verbose:
            print(f"[{qi}/{len(queries)}] {q!r}: {len(found)} new slugs so far")

    return found, reasons


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links_md", type=Path, default=Path("LINKS.md"))
    p.add_argument("--out", type=Path, default=Path("/tmp/new_slugs.txt"))
    p.add_argument("--meta_out", type=Path, default=Path("/tmp/new_slugs_meta.json"))
    p.add_argument("--user_data_dir", type=Path, default=Path.home() / ".grabcad_playwright")
    p.add_argument("--target", type=int, default=800,
                   help="Stop once this many NEW STEP-bearing slugs are found")
    p.add_argument("--max_pages", type=int, default=100,
                   help="Pages per query; the API caps at 100 regardless")
    p.add_argument("--max_per_query", type=int, default=15,
                   help="Cap each query's contribution so the sweep spreads across "
                        "the machining vocabulary instead of saturating on the first "
                        "term. Keep target/max_per_query well under len(QUERIES).")
    p.add_argument("--sleep_ms", type=int, default=250,
                   help="Politeness delay between page requests; too low trips the WAF")
    p.add_argument("--append_links", action="store_true",
                   help="Append discovered slugs to LINKS.md as a new section")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    existing = load_existing(args.links_md)
    print(f"{len(existing)} slugs already in {args.links_md}; targeting {args.target} new")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(args.user_data_dir), headless=False,
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()
        # A real page load first: the WAF challenge is solved by the browser and
        # the resulting cookies are what let context.request through at all.
        page.goto("https://grabcad.com/library", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        found, reasons = harvest(ctx.request, QUERIES, existing, args.target,
                                 args.max_pages, args.sleep_ms, not args.quiet,
                                 args.max_per_query)
        ctx.close()

    slugs = sorted(found)
    args.out.write_text("\n".join(slugs) + ("\n" if slugs else ""))
    args.meta_out.write_text(json.dumps(found, indent=2))

    print("\n" + "=" * 60)
    print(f"NEW slugs found: {len(slugs)}")
    print("filter/skip reasons:")
    for r, n in reasons.most_common(12):
        print(f"   {r:24s} {n}")
    by_q = collections.Counter(v["query"] for v in found.values())
    print("\ntop contributing queries:")
    for q, n in by_q.most_common(8):
        print(f"   {n:4d}  {q}")
    print(f"\nslug list: {args.out}\nmetadata:  {args.meta_out}")

    if args.append_links and slugs:
        with args.links_md.open("a") as fh:
            fh.write(
                "\n\n## Harvested via community API "
                f"(harvest_grabcad_slugs.py, {len(slugs)} slugs)\n\n"
                "Filtered to models publishing a neutral CAD format (STEP/IGES/\n"
                "Parasolid); mesh-only and obvious assemblies excluded. STEP is not\n"
                "guaranteed — the GrabCAD tag covers IGES too — so the downloader\n"
                "still verifies and geometric curation still applies.\n\n"
            )
            for s in slugs:
                fh.write(f"- https://grabcad.com/library/{s}\n")
        print(f"appended {len(slugs)} slugs to {args.links_md}")
    elif not args.append_links:
        print("(pass --append_links to write them into LINKS.md)")


if __name__ == "__main__":
    main()
