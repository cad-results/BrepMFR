"""Build ``brepformer/data/examples/``: a few samples from every local corpus.

The full corpora are gitignored (tens of GB). This copies a handful of matched
samples per folder, keeping each folder's own layout, so a fresh clone can run
preprocessing, training, inference and the label tools end to end:

    python brepformer/data/make_examples.py            # rebuild examples/
    python brepformer/data/make_examples.py --per_split 1

Raw corpora keep ``--n`` models whose step/graph/label files all exist.
Preprocessed corpora keep ``--per_split`` pickles per split, renumbered from
00000, with ``metadata.json`` / ``model_id_index.json`` / ``provenance.json``
rewritten to match. Symlinks (trial6* farms, pcq5/graphs) are dereferenced.
Samples are taken from the low end of the size distribution, not the minimum,
so they stay small without being degenerate one-face parts.
"""

import argparse
import json
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parent
REPO = DATA.parent.parent

# Raw corpora: folder -> {subdir: filename pattern}, "{id}" is the model id.
RAW = {
    "mftrcad": {"graphs": "{id}.json", "labels": "{id}.json", "steps": "{id}.step",
                "labels_rel": "{id}_rel.json"},
    "defeature": {"graphs": "{id}.json", "labels": "{id}.json", "steps": "{id}.step"},
    "defeature_uv": {"graphs": "{id}.json", "labels": "{id}.json"},
    "defeature_uv0": {".": "{id}.json"},
    "suitable_samples_pcq": {"labels": "{id}_result.json", "steps": "{id}.step"},
    "suitable_samples_pcq5": {"graphs": "{id}.json", "labels": "{id}_result.json", "steps": "{id}.step"},
    "suitable_samples_pcq27": {"graphs": "{id}.json", "labels": "{id}_result.json", "steps": "{id}.step"},
    "suitable_samples_scaled": {"labels": "{id}_result.json", "steps": "{id}.step"},
}
PROCESSED = [
    "defeature_processed",
    "defeature_uv_processed",
    "mftrcad_processed",
    "suitable_samples_pcq5_processed",
    "suitable_samples_pcq27_processed",
    "trial6_processed",
    "trial6a_processed",
]
STEP_SUFFIXES = {".step", ".stp"}
MAX_FILE = 8 * 1024 * 1024  # skip any single file larger than this


def pick(items, n, size):
    """``n`` items from the 10th-40th size percentile, smallest first."""
    items = sorted((i for i in items if size(i) <= MAX_FILE), key=size)
    if len(items) <= n:
        return items
    lo, hi = int(len(items) * 0.1), max(int(len(items) * 0.4), int(len(items) * 0.1) + n)
    window = items[lo:hi]
    step = max(len(window) // n, 1)
    return window[::step][:n]


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src.resolve(), dst)  # resolve(): dereference symlink farms


def build_raw(name, layout, out, n):
    src = DATA / name
    # "labels_rel" is optional (mftrcad ships it for most models, not all).
    required = {k: v for k, v in layout.items() if k != "labels_rel"}
    anchor_dir, anchor_pat = next(iter(required.items()))
    suffix = anchor_pat.replace("{id}", "")
    ids = [p.name[: -len(suffix)] for p in (src / anchor_dir).iterdir()
           if p.name.endswith(suffix) and not p.name.endswith("_rel" + suffix)]

    def paths(mid):
        return {k: src / ("labels" if k == "labels_rel" else k) / pat.format(id=mid)
                for k, pat in layout.items()}

    complete = [m for m in ids if all(p.exists() for k, p in paths(m).items() if k in required)]
    chosen = pick(complete, n, lambda m: sum(p.stat().st_size for p in paths(m).values() if p.exists()))
    for mid in chosen:
        for k, p in paths(mid).items():
            if p.exists():
                copy(p, out / name / p.relative_to(src))
    if (src / "dataset_info.json").exists():
        copy(src / "dataset_info.json", out / name / "dataset_info.json")
    return chosen


def build_processed(name, out, per_split):
    src, dst = DATA / name, out / name
    meta = json.loads((src / "metadata.json").read_text())
    index_path, prov_path = src / "model_id_index.json", src / "provenance.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else None
    # model_id_index maps id -> absolute path of the pkl; invert it to keep ids.
    by_path = {p: mid for mid, p in (index or {}).items()}

    new_index, new_prov, sizes = {}, {}, {}
    for split in ("train", "val", "test"):
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        pkls = [p for p in split_dir.iterdir() if p.suffix == ".pkl"]
        chosen = pick(pkls, per_split, lambda p: p.resolve().stat().st_size)
        width = len(pkls[0].stem) if pkls else 5
        sizes[split] = len(chosen)
        new_prov[split] = {}
        for i, p in enumerate(sorted(chosen)):
            target = dst / split / f"{i:0{width}d}.pkl"
            copy(p, target)
            mid = by_path.get(str(p), by_path.get(str(p.resolve())))
            if mid is not None:
                new_index[mid] = str(target.relative_to(REPO))
            if prov is not None:
                new_prov[split][target.stem] = prov[split].get(p.stem)

    meta["num_samples"] = sum(sizes.values())
    meta["split_sizes"] = sizes
    meta["example_subset_of"] = f"brepformer/data/{name}"
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if index is not None:
        (dst / "model_id_index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    if prov is not None:
        (dst / "provenance.json").write_text(json.dumps(new_prov, indent=2) + "\n")
    return sizes


def build_hard_samples(out, n):
    src, dst = DATA / "hard_samples", out / "hard_samples"
    steps = [p for p in src.iterdir() if p.suffix.lower() in STEP_SUFFIXES]
    for p in pick(steps, n, lambda p: p.stat().st_size):
        copy(p, dst / p.name)
    copy(src / "PLAN.md", dst / "PLAN.md")
    # Triage buckets: one pseudo-labelled model each (step + .json/.seg labels).
    kept = {}
    for bucket in ("yes", "maybe", "no"):
        (dst / bucket / "labels").mkdir(parents=True, exist_ok=True)
        (dst / bucket / "steps").mkdir(parents=True, exist_ok=True)
        bsteps = [p for p in (src / bucket / "steps").iterdir() if p.suffix.lower() in STEP_SUFFIXES]
        chosen = pick(bsteps, 1, lambda p: p.stat().st_size)
        for p in chosen:
            copy(p, dst / bucket / "steps" / p.name)
            for lab in (src / bucket / "labels").glob(p.stem + ".*"):
                copy(lab, dst / bucket / "labels" / lab.name)
        kept[bucket] = [p.stem for p in chosen]
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "examples")
    ap.add_argument("--n", type=int, default=3, help="models per raw corpus")
    ap.add_argument("--per_split", type=int, default=2, help="pickles per split per processed corpus")
    args = ap.parse_args()

    # Clear previous examples but keep the hand-written README.
    if args.out.exists():
        for p in args.out.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
    for name, layout in RAW.items():
        if (DATA / name).is_dir():
            print(f"{name}: {build_raw(name, layout, args.out, args.n)}")
    for name in PROCESSED:
        if (DATA / name / "metadata.json").exists():
            print(f"{name}: {build_processed(name, args.out, args.per_split)}")
    if (DATA / "hard_samples").is_dir():
        print(f"hard_samples: {build_hard_samples(args.out, args.n)}")
    if (DATA / "inference").is_dir():
        steps = [p for p in (DATA / "inference").iterdir() if p.suffix.lower() in STEP_SUFFIXES]
        for p in pick(steps, args.n, lambda p: p.stat().st_size):
            copy(p, args.out / "inference" / p.name)
        print(f"inference: {sorted(p.name for p in (args.out / 'inference').iterdir())}")
    # sample/ is already tracked; its processed form is three small split pickles.
    if (DATA / "sample_processed").is_dir():
        for p in (DATA / "sample_processed").iterdir():
            copy(p, args.out / "sample_processed" / p.name)
        print("sample_processed: copied whole")


if __name__ == "__main__":
    main()
