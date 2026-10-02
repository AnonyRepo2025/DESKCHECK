# Keyword-Based Literature Search

Reproducible, keyword-driven search for conference papers on coding-agent
scaffolds / issue resolution across three independent sources. The scripts
harvest the accepted-paper lists of the target venues, keep every paper whose
title/abstract matches the keyword filter, merge and de-duplicate the hits, and
pre-classify them into signal tiers for the manual screening pass.

Target venues: ICLR, ICML, NeurIPS (2025, 2026), ICSE and FSE research tracks
(2025, 2026). Everything that defines the search lives in `config.py`.

## Layout

```
config.py              
common.py             
search_openreview.py   source 1: ICLR/ICML/NeurIPS accepted lists (papercopilot mirror of OpenReview)
search_dblp.py         source 2: ICSE/FSE research tracks (DBLP toc, researchr fallback)
search_snowball.py     source 3: Semantic Scholar forward citations of the seed papers
merge_candidates.py    
run_search.py          
requirements.txt      
```

## Quick start

```
pip install -r requirements.txt
python3 run_search.py                  
python3 run_search.py --list-keywords   
```
## Outputs

```
<out>/raw/<channel>.jsonl         keyword-matched papers per channel, with keywords_hit
<out>/raw/<channel>_counts.json   per-venue / per-seed totals (how many listed, matched)
<out>/candidates.csv              de-duplicated union, strong tier first, screening columns blank
<out>/summary.json                PRISMA-style counts: per channel, unique papers, tier histogram
<out>/cache/                      raw source downloads (delete or --no-cache to refresh)
```

## Keyword filter

Applied case-insensitively to title + abstract (plus author keywords and TL;DR
where the source has them). It is deliberately recall-oriented:

| Group | Keywords |
|---|---|
| benchmark | swe-bench, swebench, swe bench |
| task | issue resolution, issue-resolution, resolve github issue, github issue, repository-level, repo-level |
| scaffold | software engineering agent, coding agent, code agent, software agent |
| repair | program repair, bug fix, bug-fix, automated repair, patch generation, fault localization |
| generic agent | agentic, llm agent, language model agent, autonomous agent |
