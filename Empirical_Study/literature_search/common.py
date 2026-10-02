"""Shared helpers: HTTP with backoff and on-disk caching, JSONL I/O, keyword
matching and title normalisation. Imported by every search_*.py script."""
import json
import pathlib
import re
import time

import requests

import config

UA = {"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
                     "literature-search/1.0")}

RETRY_STATUSES = (429, 500, 502, 503, 504)


class SearchError(RuntimeError):
    """A source could not be read (network, anti-bot page, bad payload)."""


# ---------------------------------------------------------------- keywords
_KW_RE = None
_KEYWORDS = list(config.KEYWORDS)


def set_keywords(keywords):
    """Replace the keyword list (used by run_search.py --keywords-file)."""
    global _KW_RE, _KEYWORDS
    _KEYWORDS = [k.strip() for k in keywords if k and k.strip()]
    _KW_RE = None


def keywords():
    return list(_KEYWORDS)


def _kw_re():
    global _KW_RE
    if _KW_RE is None:
        _KW_RE = re.compile("|".join(re.escape(k) for k in _KEYWORDS),
                            re.IGNORECASE)
    return _KW_RE


def kw_match(*texts):
    """Return the sorted set of keywords (lower-cased) found in the texts."""
    blob = " ".join(_as_text(t) for t in texts)
    return sorted(set(m.group(0).lower() for m in _kw_re().finditer(blob)))


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    return str(value)


def norm_title(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def strip_tags(fragment):
    import html as htmllib
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", htmllib.unescape(text)).strip()


# -------------------------------------------------------------------- HTTP
def http_get(url, params=None, headers=None, retries=5, timeout=60):
    """GET with exponential backoff on connection errors, 429 and 5xx."""
    delay, last = 2, None
    hdrs = {**UA, **(headers or {})}
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in RETRY_STATUSES:
            last = SearchError(f"HTTP {r.status_code} from {url}")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        raise SearchError(f"HTTP {r.status_code} from {url}")
    raise SearchError(f"gave up after {retries} attempts: {url} ({last})")


def get_json(url, params=None, headers=None, **kw):
    r = http_get(url, params=params, headers=headers, **kw)
    try:
        return r.json()
    except ValueError:
        ctype = r.headers.get("content-type", "?")
        hint = ""
        if "html" in ctype.lower():
            hint = " -- looks like an HTML (anti-bot challenge?) page"
        raise SearchError(f"{url} did not return JSON (content-type {ctype}){hint}")


def cached_json(cache_path, fetch, use_cache=True):
    """Return fetch() result, memoised as JSON at cache_path."""
    cache_path = pathlib.Path(cache_path)
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except ValueError:
            cache_path.unlink()          # stale / truncated cache file
    data = fetch()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False))
    return data


# ---------------------------------------------------------------- file I/O
def write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}")


def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def raw_dir(out_dir):
    p = pathlib.Path(out_dir) / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(out_dir):
    p = pathlib.Path(out_dir) / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_common_args(parser):
    parser.add_argument("--out", default="output",
                        help="output directory (default: ./output)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached source downloads and refetch")
    return parser
