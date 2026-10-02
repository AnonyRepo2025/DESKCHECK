#!/usr/bin/env python3
"""Channel 1 -- ML venues (ICLR / ICML / NeurIPS) via the papercopilot/paperlists
mirror of the OpenReview accepted lists.

OpenReview's API v2 answers anonymous clients with an interactive anti-bot
challenge, so the community-maintained JSON dumps (title, abstract, status,
author keywords, TL;DR, GitHub link) are used instead. A paper is accepted when
its status contains an accept word (config.ACCEPT_WORDS) and no reject word. Title, abstract, keywords and TL;DR
are keyword-filtered.

Output: <out>/raw/openreview.jsonl and <out>/raw/openreview_counts.json
"""
import argparse
from collections import Counter

import requests

import config
from common import (SearchError, UA, add_common_args, cache_dir, cached_json,
                    kw_match, raw_dir, write_json, write_jsonl)

RAW_URL = "https://raw.githubusercontent.com/papercopilot/paperlists/main/{p}.json"
LFS_URL = ("https://media.githubusercontent.com/media/papercopilot/paperlists/"
           "main/{p}.json")


def fetch_list(path):
    """Download one venue list, following the Git-LFS pointer if present."""
    last_err = None
    for url in (RAW_URL.format(p=path), LFS_URL.format(p=path)):
        for _ in range(3):                 # large files occasionally truncate
            r = requests.get(url, headers=UA, timeout=600)
            if r.status_code == 404:
                last_err = SearchError(f"{url} -> 404 (list not published yet?)")
                break
            if r.status_code != 200:
                last_err = SearchError(f"{url} -> HTTP {r.status_code}")
                break
            if r.text.startswith("version https://git-lfs"):
                break                      # LFS pointer; try the media URL
            try:
                return r.json()
            except ValueError as e:
                last_err = e
    raise SearchError(f"could not fetch {path}: {last_err}")


def is_accepted(status):
    s = (status or "").strip().lower()
    return (any(w in s for w in config.ACCEPT_WORDS)
            and not any(w in s for w in config.REJECT_WORDS))


def harvest_venue(label, path, out_dir, use_cache=True):
    cache = cache_dir(out_dir) / ("paperlists_" + path.replace("/", "_") + ".json")
    papers = cached_json(cache, lambda: fetch_list(path), use_cache)
    status_counts = Counter((p.get("status") or "?").strip() for p in papers)
    accepted = [p for p in papers if is_accepted(p.get("status"))]
    rows = []
    for p in accepted:
        kws = kw_match(p.get("title"), p.get("abstract"),
                       p.get("keywords"), p.get("tldr"))
        if not kws:
            continue
        rows.append({
            "source": "openreview",
            "venue": label,
            "status": p.get("status"),
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "authors": p.get("author"),
            "openreview_id": p.get("id"),
            "url": p.get("site") or (f"https://openreview.net/forum?id={p.get('id')}"
                                     if p.get("id") else None),
            "github": p.get("github"),
            "keywords_hit": kws,
        })
    counts = {"listed_total": len(papers), "accepted_total": len(accepted),
              "keyword_matched": len(rows), "statuses": dict(status_counts)}
    return rows, counts


def run(out_dir="output", use_cache=True, venues=None):
    venues = venues or config.OPENREVIEW_VENUES
    matched, counts = [], {}
    for label, path in venues.items():
        try:
            rows, c = harvest_venue(label, path, out_dir, use_cache)
        except Exception as e:                       # one venue must not kill the run
            print(f"  {label}: FAILED ({e})")
            counts[label] = {"error": str(e)}
            continue
        matched.extend(rows)
        counts[label] = c
        print(f"  {label}: {c['listed_total']} listed, {c['accepted_total']} accepted, "
              f"{c['keyword_matched']} keyword-matched")
    write_jsonl(raw_dir(out_dir) / "openreview.jsonl", matched)
    write_json(raw_dir(out_dir) / "openreview_counts.json", counts)
    return counts


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    run(args.out, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
