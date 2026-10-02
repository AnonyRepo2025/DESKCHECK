
from __future__ import annotations
import argparse, csv, json, os, shutil
from pathlib import Path

from find_code_tokens_in_prose_icat import load_difficulties_file_counts

RQ1_DIR = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("TRAJ_ROOT", "empirical_study_trajectory"))
INPUTS = RQ1_DIR / "data" / "inputs"
GOLD_COUNTS = RQ1_DIR / "data" / "gold_patch_file_analysis.csv"


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    n = sum(1 for v in obj.values() if v.get("resolved")) if obj and isinstance(next(iter(obj.values())), dict) else "-"
    print(f"  wrote {path}  ({len(obj)} instances, resolved={n})")


def mini_verified(d: Path, out: Path) -> None:
    shutil.copy(d / "per_instance_details.json", out)
    print(f"  copied {out}")


def mini_pro(d: Path, out: Path) -> None:
    raw = json.load(open(d / "eval_results.json"))
    _dump(out, {k: {"resolved": bool(v)} for k, v in raw.items()})


def openhands(d: Path, out: Path) -> None:
    run = d / "run"
    rep = run / "output.report.json"
    if not rep.exists():
        rep = run / "report.json"
    r = json.load(open(rep))
    res = {i: {"resolved": True} for i in r.get("resolved_ids", [])}
    res.update({i: {"resolved": False} for i in r.get("unresolved_ids", [])})
    for i in r.get("empty_patch_ids", []) + r.get("error_ids", []) + r.get("incomplete_ids", []):
        res.setdefault(i, {"resolved": False})
    _dump(out, res)


def sonar(d: Path, out: Path) -> None:
    r = json.load(open(d / "results" / "results.json"))
    resolved = set(r.get("resolved", []))
    ids = [p.stem for p in (d / "trajs").glob("*.json")]
    _dump(out, {i: {"resolved": i in resolved} for i in ids})


def icat_verified(d: Path, out: Path) -> None:
    shutil.copy(d / "summary.csv", out)
    print(f"  copied {out}")


def icat_pro(d: Path, out: Path) -> None:
    rows = []
    for inst in sorted(p for p in d.iterdir() if p.name.startswith("instance_")):
        f = inst / "eval" / "eval_results.json"
        if not f.exists():
            continue
        v = json.load(open(f)).get(inst.name)
        rows.append((inst.name, 1 if v else 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["instance_id", "resolved"]); w.writerows(rows)
    print(f"  wrote {out}  ({len(rows)} instances, resolved={sum(r for _, r in rows)})")


def claudecode(d: Path, out: Path) -> None:
    with open(d / "resolution.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    _dump(out, {r["instance_id"].strip(): {"resolved": r["status"].strip().lower() == "resolved"}
                for r in rows if r.get("instance_id")})


def experepair_timelines(d: Path, out: Path, leaderboard: Path) -> None:
    if not leaderboard.exists():
        print(f"  SKIP: leaderboard results file missing: {leaderboard}")
        return
    resolved = set(json.load(open(leaderboard)).get("resolved", []))
    ids = [p.name.replace(".timeline.json", "") for p in d.glob("*.timeline.json")]
    _dump(out, {i: {"resolved": i in resolved} for i in sorted(ids)})


def lite_difficulty(out: Path) -> None:
    """Official Verified difficulty for the Lite instances that overlap Verified."""
    try:
        from datasets import load_dataset
        from find_code_tokens_in_prose import _normalize_difficulty
        lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        ver = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP lite_difficulty.json: {exc}")
        return
    vd = {r["instance_id"]: _normalize_difficulty(r["difficulty"]) for r in ver}
    diff = {r["instance_id"]: vd[r["instance_id"]] for r in lite
            if r["instance_id"] in vd and vd[r["instance_id"]]}
    json.dump(diff, open(out, "w"), indent=1, sort_keys=True)
    print(f"== wrote {out} ({len(diff)}/{len(lite)} Lite instances carry an official Verified difficulty)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=INPUTS)
    ap.add_argument("--gold-counts", type=Path, default=GOLD_COUNTS)
    ap.add_argument("--experepair-leaderboard", type=Path,
                    default=INPUTS / "experepair_lite_results.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in a.root.iterdir() if p.is_dir()):
        n = d.name
        print(f"== {n}")
        if n.startswith("minisweagent_") and n.endswith("_verified"):
            mini_verified(d, a.out / f"{n}.resolutions.json")
        elif n.startswith("minisweagent_") and n.endswith("_pro"):
            mini_pro(d, a.out / f"{n}.resolutions.json")
        elif n.startswith("openhands_"):
            openhands(d, a.out / f"{n}.resolutions.json")
        elif n.startswith("sonar-"):
            sonar(d, a.out / f"{n}.resolutions.json")
        elif n.startswith("icatAgent_") and n.endswith("_verified"):
            icat_verified(d, a.out / f"{n}.resolutions.csv")
        elif n.startswith("icatAgent_") and n.endswith("_pro"):
            icat_pro(d, a.out / f"{n}.resolutions.csv")
        elif n.startswith("claudecode_"):
            claudecode(d, a.out / f"{n}.resolutions.json")
        elif n.startswith("experepair_timelines_"):
            experepair_timelines(d, a.out / f"{n}.resolutions.json", a.experepair_leaderboard)
        else:
            print("  (skipped: unknown layout)")
    diff = load_difficulties_file_counts(a.gold_counts)
    p = a.out / "pro_difficulty.json"
    json.dump(diff, open(p, "w"), indent=1, sort_keys=True)
    print(f"== wrote {p} ({len(diff)} Pro instances, easy=1/medium=2-3/hard>=4 gold-patch files)")
    lite_difficulty(a.out / "lite_difficulty.json")


if __name__ == "__main__":
    main()
