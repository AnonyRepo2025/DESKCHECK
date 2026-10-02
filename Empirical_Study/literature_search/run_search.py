#!/usr/bin/env python3
"""Run the keyword-based literature search end to end.

    python3 run_search.py                       # all channels, ./output
    python3 run_search.py --sources snowball    # one channel
    python3 run_search.py --keywords-file my_keywords.txt --out run2
    python3 run_search.py --list-keywords       # print the active filter

Channels (each writes <out>/raw/<channel>.jsonl + <channel>_counts.json):
  openreview  ICLR / ICML / NeurIPS accepted lists (papercopilot mirror)
  dblp        ICSE / FSE research tracks (DBLP toc, researchr fallback)
  snowball    Semantic Scholar forward citations of the seed papers
Then merge_candidates.py produces <out>/candidates.csv and <out>/summary.json.
"""
import argparse
import pathlib
import sys
import time

import config
import common
import merge_candidates
import search_dblp
import search_openreview
import search_snowball

SOURCES = {
    "openreview": lambda a: search_openreview.run(a.out, use_cache=not a.no_cache),
    "dblp": lambda a: search_dblp.run(a.out, use_cache=not a.no_cache),
    "snowball": lambda a: search_snowball.run(a.out, use_cache=not a.no_cache,
                                              years=a.years),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap)
    ap.add_argument("--sources", nargs="+", choices=sorted(SOURCES),
                    default=sorted(SOURCES), help="channels to run (default: all)")
    ap.add_argument("--years", type=int, nargs="+", default=list(config.YEARS),
                    help="publication years kept by the snowball channel")
    ap.add_argument("--keywords-file", type=pathlib.Path,
                    help="text file with one keyword per line; replaces config.KEYWORDS")
    ap.add_argument("--skip-merge", action="store_true",
                    help="only harvest; do not rebuild candidates.csv")
    ap.add_argument("--list-keywords", action="store_true",
                    help="print the active keyword list and exit")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # progress visible when redirected

    if args.keywords_file:
        common.set_keywords(args.keywords_file.read_text().splitlines())
    if args.list_keywords:
        print("\n".join(common.keywords()))
        return 0

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"keywords ({len(common.keywords())}): {', '.join(common.keywords())}")
    failed = []
    for name in args.sources:
        print(f"\n== {name} ==")
        t0 = time.time()
        try:
            counts = SOURCES[name](args)
        except Exception as e:                    # keep going with other channels
            print(f"  {name}: FAILED ({e})")
            failed.append(name)
            continue
        if counts and all(isinstance(c, dict) and "error" in c for c in counts.values()):
            failed.append(name)
        print(f"  ({time.time() - t0:.0f}s)")

    if not args.skip_merge:
        print("\n== merge ==")
        merge_candidates.run(args.out)
    if failed:
        print(f"\nWARNING: channels with no usable data: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
