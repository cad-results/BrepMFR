"""Would an integer PLANE-COUNT fact fix the 0%-recall polygonal classes?

geo_label scores 0% on triangular_passage / 6sides_passage / rectangular_passage and 18-33%
on the pocket variants -- together ~33% of all faces. Two existing facts nearly carry the
answer and both throw it away:
  strict_component_surfaces  collapses types to a SET, so a triangular and a hex pocket are
                             both just "plane"
  strict_component_size      bins as 0-2/2/3-5/5-9/>=9, so a rectangular pocket (4 walls +
                             floor = 5) and a hex pocket (6 + 1 = 7) share the "5-9" bin
So the side count is present in the geometry and destroyed by the binning. Test the fact at
integer resolution before paying to re-measure 12,000 models.
"""
import sys, collections, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brepformer.data.label_harness import iter_ground_truth
from brepformer.face_geometry import face_geometry
from brepformer.data.classes import CLASS_NAMES

def plane_counts(geo, n):
    comp = collections.defaultdict(list)
    types = {}
    for i in range(n):
        r = geo.get(i) or geo.get(str(i))
        if r and r.get("strict_component_id") is not None:
            comp[r["strict_component_id"]].append(i)
            types[i] = r.get("surface_type")
    out = {}
    for cid, members in comp.items():
        np_ = sum(1 for i in members if types.get(i) == "plane")
        nc = sum(1 for i in members if types.get(i) == "cylinder")
        for i in members:
            out[i] = (np_, nc)
    return out

joint = collections.Counter()
per_class = collections.Counter()
for step_path, gt in iter_ground_truth(Path('brepformer/data/mftrcad'), 60, 0, "27"):
    try:
        geo = face_geometry(str(step_path), with_topology=True)
    except Exception:
        continue
    pcnt = plane_counts(geo, len(gt))
    for i, truth in enumerate(gt):
        if truth < 0 or i not in pcnt:
            continue
        np_, nc = pcnt[i]
        joint[(min(np_, 10), min(nc, 3), truth)] += 1
        per_class[truth] += 1

tot = sum(joint.values())
pc, pf = collections.Counter(), collections.Counter()
for (a, b, c), v in joint.items():
    pc[c] += v; pf[(a, b)] += v
mi = sum((v/tot) * math.log2((v/tot) / ((pc[c]/tot)*(pf[(a,b)]/tot))) for (a,b,c), v in joint.items())
hc = -sum((v/tot)*math.log2(v/tot) for v in pc.values())
print(f"{tot} faces.  I(class ; n_planes,n_cyls) = {mi:.3f} bits of H(class)={hc:.3f} "
      f"-> {100*mi/hc:.1f}% of the label entropy")

# how much does the coarse existing bin throw away? same MI with size binned as geo_label bins it
def sbin(k):
    return ">=9" if k >= 9 else "5-9" if k >= 5 else "3-5" if k >= 3 else "2" if k == 2 else "0-2"
j2 = collections.Counter()
for (a, b, c), v in joint.items():
    j2[(sbin(a+b), c)] += v
p2 = collections.Counter()
for (f, c), v in j2.items():
    p2[f] += v
mi2 = sum((v/tot)*math.log2((v/tot)/((pc[c]/tot)*(p2[f]/tot))) for (f, c), v in j2.items())
print(f"same signal through the EXISTING coarse size bin: {mi2:.3f} bits "
      f"-> {100*mi2/hc:.1f}%   (integer resolution recovers {mi-mi2:+.3f} bits)\n")

print(f"{'class':<26}{'faces':>6}   top (n_planes,n_cyls) in component")
for c, n in sorted(per_class.items(), key=lambda kv: -kv[1]):
    if n < 25:
        continue
    bins = collections.Counter()
    for (a, b, cc), v in joint.items():
        if cc == c:
            bins[(a, b)] += v
    top = "  ".join(f"({a},{b}):{100*v/n:.0f}%" for (a, b), v in bins.most_common(3))
    print(f"{CLASS_NAMES[c]:<26}{n:>6}   {top}")
