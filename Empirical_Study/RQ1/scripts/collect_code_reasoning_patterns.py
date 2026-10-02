from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_IN = Path(__file__).resolve().parent.parent.parent / "results"
SUFFIX = "_allreasoning.json"

# (reported family, detector families it covers)
FAMILIES = [
    ("output_and_exceptions", ["output", "others"]),
    ("variable_state", ["state"]),
    ("call_dependency", ["call_dep"]),
    ("conditional", ["cond"]),
    ("loop", ["loop"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN,
                    help=f"directory of *{SUFFIX} files (default: {DEFAULT_IN})")
    a = ap.parse_args()

    files = sorted(a.in_dir.glob(f"*{SUFFIX}"))
    if not files:
        raise SystemExit(f"No *{SUFFIX} files under {a.in_dir}")

    n_traj = 0
    fam_inst, fam_hits = Counter(), Counter()
    kind_inst = {label: Counter() for label, _ in FAMILIES}
    kind_hits = {label: Counter() for label, _ in FAMILIES}
    for f in files:
        r = json.load(open(f))
        n_traj += len(r["instances"])
        for label, fams in FAMILIES:
            fam_inst[label] += sum(1 for i in r["instances"] if any(i.get(f"{x}_kinds") for x in fams))
            for x in fams:
                fam_hits[label] += r.get(f"{x}_n_total_hits", 0)
                kind_inst[label].update(r.get(f"{x}_kind_instance_counts") or {})
                kind_hits[label].update(r.get(f"{x}_kind_counts") or {})

    print(f"Overall code-reasoning patterns ({n_traj} trajectories, {len(files)} sets)")
    print(f"  {'family / sub-category':42s} {'#traj':>7s} {'%traj':>6s} {'#hits':>8s}")
    for label, _ in FAMILIES:
        print(f"  {label:42s} {fam_inst[label]:7d} {100 * fam_inst[label] / n_traj:5.1f}% {fam_hits[label]:8d}")
        for kind, hits in kind_hits[label].most_common():
            print(f"    {kind:40s} {kind_inst[label][kind]:7d} {'':6s} {hits:8d}")


if __name__ == "__main__":
    main()
