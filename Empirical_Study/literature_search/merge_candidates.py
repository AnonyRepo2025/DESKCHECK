#!/usr/bin/env python3
"""Merge the raw channel outputs, de-duplicate by normalised title and
pre-classify every match into a signal tier (see config.STRONG_RE / AGENT_RE).

  strong : repair / SWE-bench / issue-resolution wording AND agent wording
           -> left for manual screening (decision column blank)
  medium : repair / SWE wording without agent wording       -> AUTO-MEDIUM
  weak   : only generic agent keywords                     -> AUTO-WEAK

Output: <out>/candidates.csv (strong rows first) and <out>/summary.json
(per-channel counts plus the tier histogram, i.e. the PRISMA numbers).
"""
import argparse
import csv
import json
import pathlib

import config
from common import norm_title, read_jsonl, write_json

CHANNELS = ("openreview", "dblp", "snowball")

FIELDS = ["tier", "title", "venues", "sources", "url", "github", "keywords_hit",
          "abstract_snippet",
          # manual screening columns
          "is_scaffold_paper", "traj_released", "traj_url", "decision", "reason"]


def tier(row):
    blob = " ".join(str(row.get(k) or "") for k in ("title", "abstract", "keywords_hit"))
    if config.STRONG_RE.search(blob) and config.AGENT_RE.search(blob):
        return "strong"
    if config.STRONG_RE.search(blob):
        return "medium"
    return "weak"


def merge(raw):
    rows = {}
    for ch in CHANNELS:
        for r in read_jsonl(raw / f"{ch}.jsonl"):
            key = norm_title(r.get("title"))
            if not key:
                continue
            entry = rows.get(key)
            if entry is None:
                entry = rows[key] = {
                    "title": r.get("title"), "abstract": "", "url": "", "github": "",
                    "sources": set(), "venues": set(), "keywords_hit": set(),
                }
            entry["sources"].add(r.get("source") or ch)
            if r.get("venue"):
                entry["venues"].add(r["venue"])
            entry["keywords_hit"].update(r.get("keywords_hit") or [])
            # keep the richest metadata seen for this title
            if len(r.get("abstract") or "") > len(entry["abstract"]):
                entry["abstract"] = r["abstract"]
            entry["url"] = entry["url"] or r.get("url") or ""
            entry["github"] = entry["github"] or r.get("github") or ""
    for e in rows.values():
        e["keywords_hit"] = ";".join(sorted(e["keywords_hit"]))
    return rows


def write_csv(rows, path):
    tiers = {"strong": 0, "medium": 0, "weak": 0}
    order = sorted(rows, key=lambda k: ({"strong": 0, "medium": 1, "weak": 2}[tier(rows[k])],
                                        rows[k]["title"] or ""))
    with pathlib.Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in order:
            r, t = rows[key], tier(rows[key])
            tiers[t] += 1
            w.writerow({
                "tier": t, "title": r["title"],
                "venues": ";".join(sorted(r["venues"])),
                "sources": ";".join(sorted(r["sources"])),
                "url": r["url"], "github": r["github"],
                "keywords_hit": r["keywords_hit"],
                "abstract_snippet": (r["abstract"] or "")[:400].replace("\n", " "),
                "is_scaffold_paper": "", "traj_released": "", "traj_url": "",
                "decision": "" if t == "strong" else "excluded",
                "reason": "" if t == "strong" else "AUTO-" + t.upper(),
            })
    return tiers


def run(out_dir="output"):
    out = pathlib.Path(out_dir)
    raw = out / "raw"
    rows = merge(raw)
    tiers = write_csv(rows, out / "candidates.csv")
    summary = {"channels": {}, "unique_papers": len(rows), "tiers": tiers,
               "keywords": config.KEYWORDS}
    for ch in CHANNELS:
        cfile = raw / f"{ch}_counts.json"
        summary["channels"][ch] = {
            "matched_rows": len(read_jsonl(raw / f"{ch}.jsonl")),
            "counts": json.loads(cfile.read_text()) if cfile.exists() else None,
        }
    write_json(out / "summary.json", summary)
    print(f"  {len(rows)} unique papers -> {out / 'candidates.csv'}")
    print(f"  tiers: {tiers}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="output")
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
