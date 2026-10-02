#!/usr/bin/env python3
"""Driver for the four-call Claude Code pipeline (one process per instance).

Usage:
    python -m simagent.claude_code.flow_run --dataset test.jsonl --out RUN_DIR <instance_id> [...]
        [--model haiku|sonnet] [--config swebench_pro.yaml] [--skip-reasoning localize,repair,audit,validate]

Writes RUN_DIR/<iid>/{1_localize,2_repair,3_audit,4_validate}.traj.json, the patches, pipeline.json,
<iid>.traj.json (combined, mini-swe-agent-1.1 shape) and merges RUN_DIR/preds.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


from simagent import pipeline as sp  # noqa: E402

from simagent.claude_code import env as ccenv, flow, net  # noqa: E402
from simagent.claude_code.phases import audit, localize, repair, repro_localize, validate  # noqa: E402

_sub = sp._sub


def _log_for(path: Path):
    def _log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with path.open("a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
    return _log


def _merge_preds(out: Path, iid: str, model: str, patch: str) -> None:
    p = out / "preds.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[iid] = {"model_name_or_path": f"ccflow-{model}", "instance_id": iid, "model_patch": patch}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, p)


def _combined_traj(ctx: flow.Ctx, phases: list[tuple[str, Path]], patch: str) -> dict:
    msgs, cost, calls = [], 0.0, 0
    for name, path in phases:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        msgs.append(sp._banner(name, f"{name} call: {len(d.get('steps', []))} steps"))
        msgs += d.get("messages", [])
        cost += d.get("cost", 0.0) or 0.0
        calls += d.get("n_calls", 0) or 0
    return {"info": {"model_stats": {"instance_cost": cost, "api_calls": calls},
                     "config": {"agent": {"agent_type": "simagent.claude_code.flow", "model": ctx.model}},
                     "exit_status": "Submitted" if patch.strip() else "NoPatch", "submission": patch},
            "messages": msgs, "trajectory_format": "mini-swe-agent-1.1", "instance_id": ctx.instance["instance_id"],
            "pipeline": {"phases": [n for n, _ in phases], "record": ctx.record}}


def run_instance(inst: dict, out: Path, model: str, base_cfg: dict, skip: set[str], baseline: bool = False, arm: str = "reason") -> dict:
    iid = inst["instance_id"]
    inst_out = out / iid
    inst_out.mkdir(parents=True, exist_ok=True)
    log = _log_for(inst_out / "run.log")
    _sub.set_lang(inst.get("repo_language") or "python")
    rec = {"instance_id": iid, "model": model, "variant": f"ccflow-{arm}", "status": "unknown"}
    t0 = time.time()
    env = None
    try:
        env = ccenv.start_env(inst, base_cfg, out / ".env")
        rp = (base_cfg.get("environment") or {}).get("cwd", "/app")
        log(f"[{iid[:40]}] container {env.container_id[:12]} claude={env._ccpipe_claude_version}")
        if _sub.is_js():
            # J1 (JS port): _JS_RUNNER is bound per instance by the stock run_pipeline only. Without it
            # _js_run_tests returns "[js runner not detected]" for every call, so the regression guard,
            # the mutation gate's executed check and the JS language note all run blind.
            try:
                sp._JS_RUNNER.clear()
                sp._JS_RUNNER.update(sp._detect_js_runner(env, rp))
                log(f"[{iid[:40]}] js runner: kind={sp._JS_RUNNER.get('kind')} word={sp._JS_RUNNER.get('word')!r} "
                    f"workspaces={len(sp._JS_RUNNER.get('workspaces') or [])}")
                rec["js_runner"] = {k: sp._JS_RUNNER.get(k) for k in ("kind", "word", "has_yarn", "entry")}
            except Exception as e:  # noqa: BLE001
                log(f"[{iid[:40]}] js runner detection FAILED: {type(e).__name__}: {e}")
                rec["js_runner"] = {"error": f"{type(e).__name__}: {e}"}
        # Runtime artefacts the agents create that no fix contains: Redis RDB snapshots (NodeBB agents start
        # redis-server in the repo root; dump.rdb leaked into 17 JS batch-1 patches, both arms). Excluding
        # them in .git/info/exclude hides them from `git add -A -N` / `git diff HEAD` in every extraction.
        try:
            env.execute({"command": f"cd {rp} && mkdir -p .git/info && printf '%s\\n' 'dump.rdb' '*.rdb' >> .git/info/exclude"},
                        timeout=30)
        except Exception:
            pass
        try:
            dirt = env.execute({"command": f"cd {rp} && git status --porcelain"}, timeout=60)
            env._pipeline_start_dirt = [l[3:].strip() for l in (dirt.get("output") or "").splitlines() if l[:2].strip()][:60]
        except Exception:
            env._pipeline_start_dirt = []
        ctx = flow.Ctx(env=env, repo_path=rp, instance=inst, inst_out=inst_out, model=model, log=log,
                       effort=os.getenv("CCPIPE_EFFORT") or None, problem_statement=inst["problem_statement"],
                       baseline=baseline)
        sp.BUDGET.reset(float(os.getenv("INSTANCE_COST_CAP", "0") or 0))
        if arm == "repro":
            repro_localize.run(ctx)
            ctx.baseline = True          # repair without lenses, validate without the table
            repair.run(ctx)
            validate.run(ctx)
        else:
            if "localize" not in skip:
                localize.run(ctx)
            if ctx.findings is None:
                ctx.findings = _sub.SubAgentFindings()
                ctx.findings.site_reasons = {}
            repair.run(ctx)
            if "audit" not in skip:
                audit.run(ctx)
            if "validate" not in skip:
                validate.run(ctx)
        patch = ctx.patches.get("final") or sp._extract_patch(env, rp)
        if not patch.strip():
            patch = ctx.patches.get("audited") or ctx.patches.get("refined") or ctx.patches.get("initial") or ""
        (inst_out / "model_patch.diff").write_text(patch)
        rec.update(status="Completed", patch_chars=len(patch), cost=round(ctx.cost, 4), duration=round(time.time() - t0),
                   phases=ctx.record, patches={k: len(v) for k, v in ctx.patches.items()},
                   # patch texts, so metrics/emit_best_sites.py (and any pipeline.json consumer) can score
                   # the sites the patch edits against the localize sites without reading the diff files
                   model_patch=patch, initial_patch=ctx.patches.get("initial", ""),
                   refined_patch=ctx.patches.get("refined", ""))
        traj = _combined_traj(ctx, [("localize", inst_out / "1_localize.traj.json"), ("repair", inst_out / "2_repair.traj.json"),
                                    ("audit", inst_out / "3_audit.traj.json"), ("validate", inst_out / "4_validate.traj.json")], patch)
        (inst_out / f"{iid}.traj.json").write_text(json.dumps(traj, indent=1, default=str))
        _merge_preds(out, iid, model, patch)
        log(f"[{iid[:40]}] DONE patch={len(patch)}B cost=${ctx.cost:.2f} ({time.time()-t0:.0f}s)")
    except Exception as e:  # noqa: BLE001
        rec.update(status=f"HarnessError:{type(e).__name__}", error=str(e)[:500], traceback=traceback.format_exc()[-3000:])
        log(f"[{iid[:40]}] FAILED {type(e).__name__}: {e}")
    finally:
        if env is not None:
            ccenv.stop_env(env)
        (inst_out / "pipeline.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_ids", nargs="+")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=os.getenv("CCPIPE_MODEL", "sonnet"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--skip-reasoning", default="", help="comma list: localize,audit,validate (ablation)")
    ap.add_argument("--baseline", action="store_true", help="arm B: same four calls and gates, no code-reasoning / execution-simulation steps")
    ap.add_argument("--arm", choices=("reason", "baseline", "repro"), default=None,
                    help="reason (default) | baseline (= --baseline) | repro: localize by writing reproduction tests -> repair -> validate by tests (no audit, no reasoning steps)")
    args = ap.parse_args(argv)
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in args.dataset.open() if l.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    base_cfg = ccenv.load_base_config(args.config)
    ccenv.auth_env()
    net.ensure_proxy()
    skip = {x.strip() for x in args.skip_reasoning.split(",") if x.strip()}
    for iid in args.instance_ids:
        if iid not in rows:
            print(f"!! unknown instance {iid}", flush=True)
            continue
        arm = args.arm or ("baseline" if args.baseline else "reason")
        run_instance(rows[iid], args.out, args.model, base_cfg, skip, baseline=(arm == "baseline"), arm=arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
