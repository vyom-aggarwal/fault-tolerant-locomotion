import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p  # noqa: F401  (DLL load order on Windows)
from stable_baselines3 import PPO

from envs.quadruped_env import make_env_from_model_path
from amplitude_adapter import AmplitudeAdapter
from baseline_fault_eval import (analyze_trace, FAULT_ONSET_MIN, FAULT_ONSET_MAX,
                                 POST_FAULT_WINDOW)

FAULTS = [
    ("torque_limit", 0.2),
    ("joint_lock", 1.0),
    ("actuation_delay", 5),
    ("sensor_dropout", 1.0),
    ("sensor_noise", 0.3),
]
MODES = ["none", "fixed", "adaptive"]


def run_trial(model, env, adapter, fault, severity, seed):
    adapter.reset()
    obs, info = env.reset(seed=seed)
    onset = int(np.random.default_rng(seed).integers(FAULT_ONSET_MIN, FAULT_ONSET_MAX))

    pre = []
    for _ in range(onset):
        adapter.apply(env)
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        adapter.observe(info)
        pre.append(info["forward_vel"])
        if term or trunc:
            return None

    env.trigger_fault(fault, severity=severity)

    post, fell = [], False
    for _ in range(POST_FAULT_WINDOW):
        adapter.apply(env)
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        adapter.observe(info)
        post.append(info["forward_vel"])
        if term:
            fell = True
            break
        if trunc:
            return None          # window truncated; unusable

    m = analyze_trace(pre, post, fell)
    d = adapter.diagnostics()
    return {
        "fault_type": fault, "severity": severity, "mode": adapter.mode, "seed": seed,
        "baseline_vel": round(m["baseline_vel"], 5),
        "degraded": m["degraded"], "recovery_status": m["recovery_status"],
        "recovery_time_s": round(m["recovery_time_s"], 4) if m["recovery_time_s"] is not None else "",
        "settled_vel": round(m["settled_vel"], 5) if m["settled_vel"] is not None else "",
        "settled_deficit": round(m["settled_deficit"], 5) if m["settled_deficit"] is not None else "",
        "fell": fell,
        "mean_scale": d["mean_scale"], "min_scale": d["min_scale"],
        "backoff_steps": d["backoff_steps"], "threshold": d["threshold"],
    }


def evaluate(model_path, trials, fixed_scale):
    model = PPO.load(model_path)
    env = make_env_from_model_path(model_path, render=False)
    base_scale = float(env._action_scale)
    rows = []
    for fault, sev in FAULTS:
        line = f"  {fault}({sev}): "
        for mode in MODES:
            adapter = AmplitudeAdapter(base_scale=base_scale, mode=mode,
                                       fixed_scale=fixed_scale)
            n_fell = n_rec = n_ok = 0
            for sd in range(trials):
                row = run_trial(model, env, adapter, fault, sev, sd)
                env._action_scale = base_scale          # adapter mutates it
                if row is None:
                    continue
                row["model"] = os.path.basename(model_path)
                rows.append(row)
                n_ok += 1
                n_fell += bool(row["fell"])
                n_rec += (row["recovery_status"] == "recovered")
            line += (f"{mode} {n_fell}/{n_ok} fell, {n_rec} rec   ")
        print(line)
    env.close()
    return rows


def summarize(rows, fixed_scale):
    by = {}
    for r in rows:
        by.setdefault((f"{r['fault_type']}({r['severity']})", r["mode"]), []).append(r)

    print("\n" + "=" * 86)
    print("ADAPTIVE vs FIXED vs NONE")
    print("=" * 86)
    print(f"  fixed scale = {fixed_scale}\n")
    print(f"  {'fault':<17}{'mode':>9}{'n':>5}{'fall rate':>11}{'recovery':>10}"
          f"{'settled v':>11}{'mean s':>8}")
    faults = sorted({k[0] for k in by})
    verdicts = {}
    for fault in faults:
        stats = {}
        for mode in MODES:
            g = by.get((fault, mode), [])
            if not g:
                continue
            n = len(g)
            fr = sum(bool(r["fell"]) for r in g) / n
            deg = [r for r in g if str(r["degraded"]).lower() == "true"]
            rr = (sum(r["recovery_status"] == "recovered" for r in deg) / len(deg)
                  if deg else float("nan"))
            sv = [float(r["settled_vel"]) for r in g if r["settled_vel"] != ""]
            sv_m = float(np.mean(sv)) if sv else float("nan")
            ms = [r["mean_scale"] for r in g if r["mean_scale"] is not None]
            stats[mode] = (n, fr, rr, sv_m, float(np.mean(ms)) if ms else float("nan"))
            print(f"  {fault if mode==MODES[0] else '':<17}{mode:>9}{n:>5}"
                  f"{fr:>10.0%}{rr:>10.0%}{sv_m:>11.4f}{stats[mode][4]:>8.3f}")
        verdicts[fault] = stats
        print()

    print("=" * 86)
    print("READING")
    print("=" * 86)
    for fault, st in verdicts.items():
        if not all(m in st for m in MODES):
            continue
        f_none, f_fix, f_ad = st["none"][1], st["fixed"][1], st["adaptive"][1]
        v_none, v_fix, v_ad = st["none"][3], st["fixed"][3], st["adaptive"][3]
        s_ad = st["adaptive"][4]

        if f_none < 0.05 and f_fix < 0.05 and f_ad < 0.05:
            msg = (f"no falls in any condition -- no stability problem here. "
                   f"Beating fixed on speed ({v_ad:.3f} vs {v_fix:.3f}) only shows "
                   f"fixed is needlessly slow, not that adaptation works.")
        elif f_ad > f_fix + 0.08:
            msg = (f"WORSE than fixed on safety: {f_ad:.0%} vs {f_fix:.0%} falls "
                   f"(mean s={s_ad:.3f} vs 0.700 -- it did not back off enough). "
                   f"Speed is not a defence.")
        elif f_ad > f_none + 0.08:
            msg = (f"WORSE than no adaptation: {f_ad:.0%} vs {f_none:.0%} falls. "
                   f"The controller is hurting.")
        elif f_ad < f_fix - 0.08:
            msg = (f"safer than fixed: {f_ad:.0%} vs {f_fix:.0%} falls -- adapts "
                   f"further than the constant when the constant is not enough")
        elif v_ad > v_fix + 0.01:
            msg = (f"same safety ({f_ad:.0%} vs {f_fix:.0%} falls) at higher speed "
                   f"({v_ad:.3f} vs {v_fix:.3f} m/s) -- adaptation earns its keep")
        else:
            msg = (f"matches fixed on both ({f_ad:.0%} falls, {v_ad:.3f} m/s) -- "
                   f"no advantage over walking conservatively")
        print(f"  {fault:<17}{msg}")
        print(f"  {'':17}(no adaptation: {f_none:.0%} falls)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--fixed_scale", type=float, default=0.70)
    parser.add_argument("--faults", type=str, default=None,
                        help="Comma-separated fault types to evaluate.")
    parser.add_argument("--severities", type=str, default=None,
                        help="Comma-separated severities, used with a single "
                             "--faults value. A constant tuned for ONE operating "
                             "point is close to unbeatable at that point; the "
                             "claim that adaptation works requires a RANGE no "
                             "single constant handles well.")
    parser.add_argument("--out", type=str, default="logs/amplitude_eval.csv")
    args = parser.parse_args()

    paths = []
    if args.all:
        mp = os.path.join(args.results_dir, "manifest.json")
        if not os.path.exists(mp):
            raise SystemExit(f"{mp} not found.")
        with open(mp) as f:
            manifest = json.load(f)
        for sid in sorted((s for s, v in manifest.get("seeds", {}).items()
                           if v.get("gait", {}).get("converged")), key=int):
            path = os.path.join(args.models_dir, f"seed_{sid}")
            if os.path.exists(path + ".zip"):
                paths.append(path)
    else:
        paths = [args.model]

    global FAULTS
    if args.faults:
        wanted = {f.strip() for f in args.faults.split(",")}
        FAULTS = [f for f in FAULTS if f[0] in wanted]
        if not FAULTS:
            raise SystemExit(f"No known fault types in {sorted(wanted)}")
    if args.severities:
        if len(FAULTS) != 1:
            raise SystemExit("--severities requires exactly one --faults value")
        name = FAULTS[0][0]
        FAULTS = [(name, float(x)) for x in args.severities.split(",")]
        print(f"Severity range for {name}: {[f[1] for f in FAULTS]}")

    all_rows = []
    for path in paths:
        print(f"\n{os.path.basename(path)}")
        all_rows += evaluate(path, args.trials, args.fixed_scale)

    if all_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        summarize(all_rows, args.fixed_scale)
        print(f"\nWrote {len(all_rows)} trials to {args.out}")


if __name__ == "__main__":
    main()
    