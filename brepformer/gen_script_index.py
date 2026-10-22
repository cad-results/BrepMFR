#!/usr/bin/env python3
"""Regenerate the "Complete Script Index" appendix at the end of SCRIPTS.md.

The index exists to answer one question cheaply: *is this script documented?* Every
runnable script in the repo gets a row, with a link to its section above if one exists
and an explicit ``undocumented`` mark if it does not. Hand-maintaining that list is how
a script ends up in the repo with no recorded invocation, so it is generated instead.

A script counts as runnable if it has an ``if __name__ == "__main__"`` block; pure
library modules (layers, configs, datasets) are listed separately and briefly, since
they are imported rather than invoked and do not need a command.

The appendix is delimited by the two marker comments below, so re-running this replaces
only that block and never touches the hand-written documentation above it.

Usage:
    python -m brepformer.gen_script_index            # rewrite the appendix
    python -m brepformer.gen_script_index --check    # exit 1 if it is stale (for CI)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "brepformer" / "SCRIPTS.md"

BEGIN = "<!-- BEGIN GENERATED SCRIPT INDEX -->"
END = "<!-- END GENERATED SCRIPT INDEX -->"

SEARCH_ROOTS = ("brepformer", "rotator")

#: Grouping is by what a reader is trying to do, not by directory, because the
#: directory layout does not track the pipeline (data acquisition lives in data/
#: alongside the preprocessing that consumes it).
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Data acquisition", ("download_grabcad", "grabcad_login", "harvest_grabcad",
                          "probe_grabcad", "curate_corpus")),
    ("Data preparation", ("prepare_defeature", "preprocess", "preprocessing",
                          "step_to_graph", "steps_to_graphs", "combine_processed",
                          "merge_suitable_samples")),
    ("Training", ("train", "train_preprocessed", "fine_tune", "trial6_sweep")),
    ("Evaluation & analysis", ("test", "test_preprocessed", "analyze", "eval_per_class",
                               "scan_checkpoints", "collect_results",
                               "validate_geometric_rules", "validate_vision_classify",
                               "validate_vision_labeling", "test_symmetry",
                               "gen_script_index")),
    ("Inference & labeling", ("infer", "reclassify_visual", "geometric_rules")),
    ("Defeaturing", ("defeature", "defeature_v2", "primitives", "compare_primitives",
                     "run_pipeline")),
    ("Visualization & export", ("visualize", "visualize_seg", "visualize_defeature",
                                "export_freecad", "export_model", "tensorboard_server",
                                "pipeline")),
]


def summary(path: Path) -> str:
    """First sentence of the module docstring, or '' if there is none."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return "_(does not parse)_"
    doc = ast.get_docstring(tree) or ""
    if not doc:
        return ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    # Keep it to one sentence; many docstrings open with a summary line then detail.
    m = re.match(r"(.+?\.)(\s|$)", first)
    return (m.group(1) if m else first)[:200]


def is_runnable(path: Path) -> bool:
    txt = path.read_text(encoding="utf-8", errors="replace")
    return '__name__ == "__main__"' in txt or "__name__ == '__main__'" in txt


def slugify(heading_text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", heading_text.lower()).strip()
    return re.sub(r"\s+", "-", slug)


def index_sections(doc_text: str) -> dict[str, tuple[str, str]]:
    """Map script filename -> (anchor, heading text) of the section that documents it.

    Scans linearly tracking the current heading, so a script counts as documented when
    it appears anywhere in a section body -- a runnable command is documentation, even
    without a heading of its own. A heading that names the script wins over a body
    mention, since that is the section actually *about* it. Everything below the
    generated appendix is ignored, otherwise the index would treat itself as the
    documentation for every script it lists.
    """
    found: dict[str, tuple[str, str, bool]] = {}
    heading = ("", "")
    for line in doc_text.splitlines():
        if line.strip() == BEGIN:
            break
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            heading = (slugify(text), text)
        # Two invocation forms count, and missing either produces a false "undocumented":
        # a path (`brepformer/data/foo.py`) and a module (`-m brepformer.data.foo`).
        names = [Path(m).name for m in re.findall(r"[\w./-]+\.py", line)]
        names += [f"{m.rsplit('.', 1)[-1]}.py"
                  for m in re.findall(r"-m\s+([\w.]+)", line)]
        for base in names:
            in_heading = line.startswith("#")
            prev = found.get(base)
            if prev is None or (in_heading and not prev[2]):
                found[base] = (heading[0], heading[1], in_heading)
    return {k: (v[0], v[1]) for k, v in found.items()}


def anchor_for(name: str, sections: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    return sections.get(name)


def collect() -> tuple[list[dict], list[dict]]:
    runnable, libs = [], []
    for root in SEARCH_ROOTS:
        for p in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in p.parts or p.name == "__init__.py":
                continue
            rel = p.relative_to(REPO)
            rec = {"path": str(rel), "name": p.name, "summary": summary(p)}
            (runnable if is_runnable(p) else libs).append(rec)
    return runnable, libs


def group_of(name: str) -> str:
    stem = Path(name).stem
    for title, keys in GROUPS:
        if stem in keys:
            return title
    return "Other"


def build() -> str:
    doc_text = DOC.read_text()
    sections = index_sections(doc_text)
    runnable, libs = collect()
    out = [BEGIN, "", "## Complete Script Index", "",
           f"Every runnable script in the repo ({len(runnable)} of "
           f"{len(runnable) + len(libs)} Python files; the rest are imported libraries, "
           "listed at the bottom). Generated by "
           "`python -m brepformer.gen_script_index` — re-run it after adding a script "
           "so nothing goes untracked. A blank **Documented** cell means the script has "
           "no recorded invocation above and needs one.", ""]

    by_group: dict[str, list[dict]] = {}
    for r in runnable:
        by_group.setdefault(group_of(r["name"]), []).append(r)

    order = [t for t, _ in GROUPS] + ["Other"]
    undocumented = []
    for title in order:
        rows = by_group.get(title)
        if not rows:
            continue
        out += [f"### {title}", "",
                "| Script | Documented in | What it does |", "|---|---|---|"]
        for r in sorted(rows, key=lambda x: x["path"]):
            a = anchor_for(r["name"], sections)
            if a:
                link = f"[{a[1]}](#{a[0]})"
            else:
                link = "**—**"
                undocumented.append(r["path"])
            out.append(f"| `{r['path']}` | {link} | {r['summary'] or '—'} |")
        out.append("")

    out += ["### Imported libraries (not invoked directly)", "",
            "| Module | What it provides |", "|---|---|"]
    for r in sorted(libs, key=lambda x: x["path"]):
        out.append(f"| `{r['path']}` | {r['summary'] or '—'} |")
    out.append("")

    if undocumented:
        out += [f"> **{len(undocumented)} script(s) have no recorded invocation above:** "
                + ", ".join(f"`{u}`" for u in undocumented) + ".", ""]

    out.append(END)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if the appendix is stale instead of rewriting it")
    args = ap.parse_args()

    text = DOC.read_text()
    block = build()
    if BEGIN in text and END in text:
        start, end = text.index(BEGIN), text.index(END) + len(END)
        new = text[:start] + block + text[end:]
    else:
        new = text.rstrip("\n") + "\n\n" + block + "\n"

    if args.check:
        if new != text:
            print("SCRIPTS.md script index is stale; run "
                  "`python -m brepformer.gen_script_index`", file=sys.stderr)
            sys.exit(1)
        print("script index is up to date")
        return

    DOC.write_text(new)
    print(f"wrote script index to {DOC} ({len(new.splitlines())} lines total)")


if __name__ == "__main__":
    main()
