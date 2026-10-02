#!/usr/bin/env python3
"""Channel 2 -- SE venues (ICSE / FSE research tracks).

Primary source: the DBLP table of contents of each volume, read through the
DBLP search API (`q=toc:<toc>:`), paginated by `f` because DBLP caps every
response at 100 hits. Fallback (used when the volume is not indexed yet or
DBLP serves its anti-bot challenge page): the "Accepted Papers" table of the
research-track page on conf.researchr.org.

Neither source exposes abstracts, so this channel keyword-filters titles only;
the snowball channel (which has abstracts) covers the resulting recall gap.

Output: <out>/raw/dblp.jsonl and <out>/raw/dblp_counts.json
"""
import argparse
import re
import time

import config
from common import (SearchError, add_common_args, cache_dir, cached_json,
                    get_json, http_get, kw_match, raw_dir, strip_tags,
                    write_json, write_jsonl)

DBLP_SEARCH = "https://dblp.org/search/publ/api"
RESEARCHR = "https://conf.researchr.org/track/{track}"


# ------------------------------------------------------------------- DBLP
def fetch_dblp_toc(toc):
    hits, first = [], 0
    while True:
        data = get_json(DBLP_SEARCH, params={"q": f"toc:{toc}:", "h": 100,
                                             "f": first, "format": "json"})
        h = data.get("result", {}).get("hits", {})
        batch = h.get("hit", []) or []
        hits.extend(batch)
        first += len(batch)
        if not batch or first >= int(h.get("@total", 0)):
            break
        time.sleep(2)
    if not hits:
        raise SearchError(f"DBLP toc {toc} returned no entries (not indexed yet?)")
    return hits


def dblp_rows(label, toc, hits):
    rows = []
    for hit in hits:
        info = hit.get("info", {})
        title = re.sub(r"\.$", "", info.get("title", "") or "")
        kws = kw_match(title)
        if not kws:
            continue
        authors = (info.get("authors") or {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        rows.append({
            "source": "dblp", "venue": label, "toc": toc,
            "title": title,
            "authors": [a.get("text") for a in authors],
            "doi": info.get("doi"), "url": info.get("ee"),
            "dblp_key": info.get("key"), "keywords_hit": kws,
        })
    return rows


# -------------------------------------------------------------- researchr
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
TITLE_RE = re.compile(r'<a href="#"[^>]*data-event-modal="[^"]*"[^>]*>(.*?)</a>', re.S)
TRACK_RE = re.compile(r'<div class="prog-track">(.*?)</div>', re.S)
AUTHOR_RE = re.compile(r'<a href="[^"]*/profile/[^"]*"[^>]*>(.*?)</a>', re.S)
LINK_RE = re.compile(r'<a href="([^"]+)"[^>]*class="publication-link[^"]*"', re.S)


def fetch_researchr(track):
    """Return [{title, authors, track, link}] for the track's accepted papers."""
    page = http_get(RESEARCHR.format(track=track), timeout=120).text
    i = page.find("<h3>Accepted Papers</h3>")
    if i < 0:
        raise SearchError("'Accepted Papers' table not found on researchr page")
    table = page[i:page.find("</table>", i)]
    papers, seen = [], set()
    for row in ROW_RE.findall(table):
        m = TITLE_RE.search(row)
        if not m:
            continue
        title = strip_tags(m.group(1))        # drops trailing artifact badges
        if not title or title in seen:
            continue
        seen.add(title)
        trk = TRACK_RE.search(row)
        lnk = LINK_RE.search(row)
        papers.append({
            "title": title,
            "authors": [strip_tags(a) for a in AUTHOR_RE.findall(row)],
            "track": strip_tags(trk.group(1)) if trk else None,
            "link": lnk.group(1) if lnk else None,
        })
    if not papers:
        raise SearchError("researchr table parsed to zero papers (markup changed?)")
    return papers


def researchr_rows(label, track, papers):
    rows = []
    for p in papers:
        kws = kw_match(p["title"])
        if not kws:
            continue
        rows.append({
            "source": "researchr", "venue": label, "track": p.get("track"),
            "title": p["title"], "authors": p.get("authors"),
            "url": p.get("link") or RESEARCHR.format(track=track),
            "keywords_hit": kws,
        })
    return rows


# -------------------------------------------------------------------- run
def harvest_venue(v, out_dir, use_cache=True):
    label = v["label"]
    errors = []
    if v.get("dblp_toc"):
        cache = cache_dir(out_dir) / f"dblp_{label.replace(' ', '')}.json"
        try:
            hits = cached_json(cache, lambda: fetch_dblp_toc(v["dblp_toc"]), use_cache)
            rows = dblp_rows(label, v["dblp_toc"], hits)
            return rows, {"source": "dblp", "toc_total": len(hits),
                          "keyword_matched": len(rows)}
        except Exception as e:
            errors.append(f"dblp: {e}")
    if v.get("researchr_track"):
        cache = cache_dir(out_dir) / f"researchr_{label.replace(' ', '')}.json"
        try:
            papers = cached_json(cache, lambda: fetch_researchr(v["researchr_track"]),
                                 use_cache)
            rows = researchr_rows(label, v["researchr_track"], papers)
            return rows, {"source": "researchr", "toc_total": len(papers),
                          "keyword_matched": len(rows),
                          "note": "; ".join(errors) or None}
        except Exception as e:
            errors.append(f"researchr: {e}")
    raise SearchError(" | ".join(errors) or "no source configured")


def run(out_dir="output", use_cache=True, venues=None):
    venues = venues or config.SE_VENUES
    matched, counts = [], {}
    for v in venues:
        try:
            rows, c = harvest_venue(v, out_dir, use_cache)
        except Exception as e:
            print(f"  {v['label']}: FAILED ({e})")
            counts[v["label"]] = {"error": str(e)}
            continue
        matched.extend(rows)
        counts[v["label"]] = c
        print(f"  {v['label']}: {c['toc_total']} papers via {c['source']}, "
              f"{c['keyword_matched']} keyword-matched (titles only)")
        time.sleep(1)
    write_jsonl(raw_dir(out_dir) / "dblp.jsonl", matched)
    write_json(raw_dir(out_dir) / "dblp_counts.json", counts)
    return counts


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args()
    run(args.out, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
