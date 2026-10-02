#!/usr/bin/env python3
"""Channel 3 -- one round of forward-citation snowballing via the Semantic
Scholar Graph API.

Every paper citing one of config.SNOWBALL_SEEDS is kept when its venue string
matches config.SNOWBALL_VENUE_RE, its year is in config.YEARS and the keyword
filter hits title or abstract. Set S2_API_KEY in the environment to lift the
anonymous rate limit (the client backs off on 429 either way).

Output: <out>/raw/snowball.jsonl and <out>/raw/snowball_counts.json
"""
import argparse
import os
import time

import config
from common import (add_common_args, cache_dir, cached_json, get_json,
                    kw_match, raw_dir, write_json, write_jsonl)

API = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,abstract,venue,year,externalIds,publicationVenue,url"


def _headers():
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else None


def fetch_citations(seed_id):
    out, offset = [], 0
    while True:
        data = get_json(f"{API}/{seed_id}/citations", headers=_headers(),
                        params={"fields": FIELDS, "limit": 1000, "offset": offset})
        batch = data.get("data", [])
        out.extend(c.get("citingPaper", {}) for c in batch)
        if data.get("next") is None or not batch:
            break
        offset = data["next"]
        time.sleep(1.5)
    return out


def venue_str(p):
    pv = p.get("publicationVenue") or {}
    return " ".join(filter(None, [p.get("venue"), pv.get("name")]))


def run(out_dir="output", use_cache=True, years=None, seeds=None):
    years = set(years or config.YEARS)
    seeds = seeds or config.SNOWBALL_SEEDS
    matched, counts, seen = [], {}, set()
    for name, sid in seeds.items():
        cache = cache_dir(out_dir) / f"s2_citations_{sid.replace(':', '_')}.json"
        try:
            cits = cached_json(cache, lambda: fetch_citations(sid), use_cache)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            counts[name] = {"error": str(e)}
            continue
        kept = venue_ok = 0
        for p in cits:
            pid = p.get("paperId")
            if not pid or pid in seen:
                continue
            v = venue_str(p)
            if not config.SNOWBALL_VENUE_RE.search(v) or p.get("year") not in years:
                continue
            venue_ok += 1
            kws = kw_match(p.get("title"), p.get("abstract"))
            if not kws:
                continue
            seen.add(pid)
            kept += 1
            ext = p.get("externalIds") or {}
            matched.append({
                "source": "snowball", "seed": name,
                "title": p.get("title"), "abstract": p.get("abstract"),
                "venue": v, "year": p.get("year"),
                "arxiv": ext.get("ArXiv"), "doi": ext.get("DOI"),
                "url": p.get("url"), "s2_id": pid, "keywords_hit": kws,
            })
        counts[name] = {"citations": len(cits), "venue_year_ok": venue_ok,
                        "keyword_matched": kept}
        print(f"  {name}: {len(cits)} citations, {venue_ok} in target venues/years, "
              f"{kept} keyword-matched (new)")
        time.sleep(2)
    write_jsonl(raw_dir(out_dir) / "snowball.jsonl", matched)
    write_json(raw_dir(out_dir) / "snowball_counts.json", counts)
    return counts


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--years", type=int, nargs="+", default=list(config.YEARS))
    args = ap.parse_args()
    run(args.out, use_cache=not args.no_cache, years=args.years)


if __name__ == "__main__":
    main()
