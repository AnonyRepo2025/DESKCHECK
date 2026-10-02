"""Search configuration: the single place that defines what is searched for.

Everything the three keyword-based channels share lives here -- the keyword
list, the signal-tier regexes used to pre-classify matches, the venues per
channel, the citation-snowball seeds and the publication years of interest.
Edit this file (or pass CLI overrides to run_search.py) to retarget the search.
"""
import re

# --------------------------------------------------------------------------
# Years of interest (applied where the source exposes a year: snowball; the
# venue lists below are already year-specific).
# --------------------------------------------------------------------------
YEARS = (2025, 2026)

# --------------------------------------------------------------------------
# Keyword filter applied case-insensitively to title + abstract (+ author
# keywords / TL;DR where available). Deliberately broad: recall over
# precision -- the tiering step and the manual screen prune false positives.
# --------------------------------------------------------------------------
KEYWORDS = [
    # benchmark
    "swe-bench", "swebench", "swe bench",
    # task
    "issue resolution", "issue-resolution", "resolve github issue",
    "github issue", "repository-level", "repo-level",
    # scaffold wording
    "software engineering agent", "coding agent", "code agent",
    "software agent",
    # repair wording
    "program repair", "bug fix", "bug-fix", "automated repair",
    "patch generation", "fault localization",
    # generic agent wording (weak signal on its own)
    "agentic", "llm agent", "language model agent", "autonomous agent",
]

# --------------------------------------------------------------------------
# Signal tiers used by merge_candidates.py to pre-classify keyword matches.
#   strong : STRONG_RE and AGENT_RE both hit  -> goes to manual screening
#   medium : STRONG_RE only (repair / SWE context, no agent wording)
#   weak   : only generic agent keywords        -> auto-excluded (AUTO-WEAK)
# Medium/weak rows stay in candidates.csv so the auto-exclusion is auditable.
# --------------------------------------------------------------------------
STRONG_RE = re.compile(
    r"swe[- ]?bench|issue[- ]resolut|resolv\w+ (real[- ]world )?(github )?issues"
    r"|program repair|automated repair|repair agent|patch generation"
    r"|repository[- ]level|repo[- ]level|fault locali[sz]|bug[- ]?fix"
    r"|software engineering agent|coding agent|code agent|software agent",
    re.IGNORECASE)
AGENT_RE = re.compile(r"agent", re.IGNORECASE)

# --------------------------------------------------------------------------
# Channel 1 -- ML venues via the papercopilot/paperlists mirror of the
# OpenReview accepted lists (OpenReview's own API serves an interactive
# anti-bot challenge to anonymous clients). label -> path inside the repo.
# --------------------------------------------------------------------------
OPENREVIEW_VENUES = {
    "ICLR 2025": "iclr/iclr2025",
    "ICLR 2026": "iclr/iclr2026",
    "ICML 2025": "icml/icml2025",
    "ICML 2026": "icml/icml2026",
    "NeurIPS 2025": "nips/nips2025",
    "NeurIPS 2026": "nips/nips2026",   # skipped gracefully if not published yet
}
# A paper counts as accepted when its lower-cased status contains one of
# these words (covers "Poster", "Oral", "Spotlight Poster",
# "ICLR 2026 ConditionalPoster", "Accept (Oral)", ...) and none of the
# rejection words.
ACCEPT_WORDS = ("poster", "spotlight", "oral", "accept")
REJECT_WORDS = ("reject", "withdraw", "desk")

# --------------------------------------------------------------------------
# Channel 2 -- SE venues. Primary source is the DBLP table of contents
# (search API, paginated); fallback is the "Accepted Papers" table of the
# track page on conf.researchr.org. Either key may be omitted. FSE research
# papers are published as Proc. ACM Softw. Eng. (PACMSE) volumes since 2024;
# the fse20xx companion volumes are deliberately not listed (workshop/demo/
# industry tracks are out of scope).
# --------------------------------------------------------------------------
SE_VENUES = [
    {"label": "ICSE 2025",
     "dblp_toc": "db/conf/icse/icse2025.bht",
     "researchr_track": "icse-2025/icse-2025-research-track"},
    {"label": "ICSE 2026",
     "dblp_toc": "db/conf/icse/icse2026.bht",
     "researchr_track": "icse-2026/icse-2026-research-track"},
    {"label": "FSE 2025",
     "dblp_toc": "db/journals/pacmse/pacmse2.bht",
     "researchr_track": "fse-2025/fse-2025-research-papers"},
    {"label": "FSE 2026",
     "dblp_toc": "db/journals/pacmse/pacmse3.bht",
     "researchr_track": "fse-2026/fse-2026-research-papers"},
]

# --------------------------------------------------------------------------
# Channel 3 -- citation snowballing (Semantic Scholar Graph API): forward
# citations of these seeds, kept when venue matches SNOWBALL_VENUE_RE, year
# is in YEARS and the keyword filter hits. One round.
# --------------------------------------------------------------------------
SNOWBALL_SEEDS = {
    "SWE-bench": "ARXIV:2310.06770",
    "SWE-agent": "ARXIV:2405.15793",
    "Agentless": "ARXIV:2407.01489",
    "AutoCodeRover": "ARXIV:2404.05427",
    "SWE-Bench-Pro": "ARXIV:2509.16941",
}
SNOWBALL_VENUE_RE = re.compile(
    r"\b(icml|iclr|neurips|neur ips|international conference on machine learning|"
    r"international conference on learning representations|"
    r"neural information processing systems|"
    r"icse|international conference on software engineering|"
    r"fse|foundations of software engineering|proc\.? acm softw)\b",
    re.IGNORECASE)
