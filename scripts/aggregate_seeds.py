import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict


def _f(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def per_seed_stats(csv_path):
    by_fault = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_fault[row["fault_type"]].append(row)

    out = {}
    for fault_type, rows in by_fault.items():
        n = len(rows)
        rec_times = [_f(r["recovery_time_s"]) for r in rows]
        recovered_times = [t for t in rec_times if t is not None]
        distances = [d for d in (_f(r["post_fault_distance_m"]) for r in rows) if d is not None]
        drops = [d for d in (_f(r.get("velocity_drop_frac")) for r in rows) if d is not None]
        settled = [d for d in (_f(r.get("settled_deficit")) for r in rows) if d is not None]
        falls = [_truthy(r["fell"]) for r in rows]
        statuses = [r.get("recovery_status", "") for r in rows]
        has_status = any(statuses)
        n_degraded = sum(1 for r in rows if _truthy(r.get("degraded", "")))
        n_recovered = sum(1 for st in statuses if st == "recovered")
        n_nodeg = sum(1 for st in statuses if st == "no_degradation")

        out[fault_type] = {
            "n_trials": n,
            "recovery_rate": (n_recovered / n_degraded) if (has_status and n_degraded)
                             else (len(recovered_times) / n if n and not has_status else 0.0),
            "degradation_rate": (n_degraded / n) if n else 0.0,
            "no_degradation_rate": (n_nodeg / n) if (has_status and n) else None,
            "fall_rate": sum(falls) / n if n else 0.0,
            "mean_velocity_drop": statistics.mean(drops) if drops else None,
            "mean_settled_deficit": statistics.mean(settled) if settled else None,
            "n_settled": len(settled),
            "mean_recovery_time_s": statistics.mean(recovered_times) if recovered_times else None,
            "mean_post_fault_distance_m": statistics.mean(distances) if distances else None,
        }
    return out


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def fmt(mean, sd, decimals=3, pct=False):
    if mean is None:
        return "n/a"
    if pct:
        base = f"{mean:.1%}"
        return base if sd is None else f"{base} ± {sd:.1%}"
    base = f"{mean:.{decimals}f}"
    return base if sd is None else f"{base} ± {sd:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="logs/across_seed_summary.csv")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "seed_*", "baseline_fault_results.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        print(f"ERROR: no per-seed CSVs found matching {pattern}")
        print("Run scripts/run_multiseed.py first.")
        return

    manifest_path = os.path.join(args.results_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    seeds_meta = manifest.get("seeds", {})
    if seeds_meta:
        kept, excluded = [], []
        for path in csv_paths:
            label = os.path.basename(os.path.dirname(path))      # "seed_3"
            num = label.split("_")[-1]
            info = seeds_meta.get(num, {})
            if info.get("gait", {}).get("converged"):
                kept.append(path)
            else:
                reason = info.get("gait", {}).get("failure_mode") or info.get("status") or "not in manifest"
                excluded.append((label, reason))

        if excluded:
            print("=" * 78)
            print("EXCLUDED FROM ANALYSIS (failed the convergence screen)")
            print("=" * 78)
            for label, reason in excluded:
                print(f"  {label}: {reason}")
            print("  These result files are stale -- from a run under a different")
            print("  criterion. They are ignored here; delete them to avoid confusion.\n")
        csv_paths = kept

        if not csv_paths:
            print("ERROR: no CONVERGED seeds have results. Nothing to analyze.")
            return
    else:
        print(f"WARNING: no manifest at {manifest_path} -- cannot verify which seeds")
        print("converged, so ALL result files are being included. Any non-converged")
        print("seed present will contaminate these numbers.\n")

    # convergence rate from the manifest, if present 
    manifest_path = os.path.join(args.results_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        seeds = manifest.get("seeds", {})
        attempted = [s for s, v in seeds.items() if v.get("status") in ("ok", "train_failed")]
        converged = [s for s, v in seeds.items()
                     if v.get("status") == "ok" and v.get("gait", {}).get("converged")]
        speeds = [seeds[s]["gait"]["mean_speed_mps"] for s in converged]
        m, sd, n = summarize(speeds)

        print("=" * 78)
        print("TRAINING RELIABILITY")
        print("=" * 78)
        print(f"  Seeds attempted        : {len(attempted)}")
        print(f"  Converged to a gait    : {len(converged)}/{len(attempted)}"
              + (f"  ({len(converged)/len(attempted):.0%})" if attempted else ""))
        print(f"  Speed (converged only) : {fmt(m, sd)} m/s  (n={n} seeds)")
        if len(converged) < len(attempted):
            print("\n  NOTE: a convergence rate below 100% is a reportable result, not a")
            print("  problem to hide. State it in the paper -- it characterizes how")
            print("  reliably the training setup produces a usable policy.")

    # per-seed, then across-seed 
    all_seeds = {}
    for path in csv_paths:
        seed_label = os.path.basename(os.path.dirname(path))
        all_seeds[seed_label] = per_seed_stats(path)

    fault_types = sorted({ft for stats in all_seeds.values() for ft in stats})
    seed_labels = sorted(all_seeds)

    print("\n" + "=" * 78)
    print(f"BASELINE A (no adaptation) -- ACROSS {len(seed_labels)} SEEDS")
    print("=" * 78)
    print("Unit of analysis is the SEED. Each seed contributes one number per")
    print("metric; the ± is the spread across seeds, not across trials.\n")

    rows_out = []

    for fault_type in fault_types:
        print(f"--- {fault_type} ---")

        rec_rates = [all_seeds[s].get(fault_type, {}).get("recovery_rate") for s in seed_labels]
        deg_rates = [all_seeds[s].get(fault_type, {}).get("degradation_rate") for s in seed_labels]
        drops = [all_seeds[s].get(fault_type, {}).get("mean_velocity_drop") for s in seed_labels]
        settled_d = [all_seeds[s].get(fault_type, {}).get("mean_settled_deficit") for s in seed_labels]
        fall_rates = [all_seeds[s].get(fault_type, {}).get("fall_rate") for s in seed_labels]
        rec_times = [all_seeds[s].get(fault_type, {}).get("mean_recovery_time_s") for s in seed_labels]
        dists = [all_seeds[s].get(fault_type, {}).get("mean_post_fault_distance_m") for s in seed_labels]

        rr_m, rr_sd, rr_n = summarize(rec_rates)
        fr_m, fr_sd, _ = summarize(fall_rates)
        rt_m, rt_sd, _ = summarize(rec_times)
        d_m, d_sd, _ = summarize(dists)

        dg_m, dg_sd, _ = summarize(deg_rates)
        dr_m, dr_sd, _ = summarize(drops)
        sd_m, sd_sd, sd_n = summarize(settled_d)
        print(f"  Degradation rate    : {fmt(dg_m, dg_sd, pct=True)}   "
              f"(fault measurably slowed the robot)")
        print(f"  Transient dip       : {fmt(dr_m, dr_sd, pct=True)}   "
              f"(worst short-window dip; descriptive only)")
        if sd_m is not None:
            print(f"  Settled deficit     : {fmt(sd_m, sd_sd)} m/s   "
                  f"(lasting loss over final 2 s; what a correction must restore)")
        print(f"  Recovery rate       : {fmt(rr_m, rr_sd, pct=True)}   "
              f"(of DEGRADED trials; n={rr_n} seeds)")
        print(f"  Fall rate           : {fmt(fr_m, fr_sd, pct=True)}")
        print(f"  Recovery time       : {fmt(rt_m, rt_sd)} s")
        print(f"  Post-fault distance : {fmt(d_m, d_sd)} m")

        per_seed_str = ", ".join(
            f"{s.replace('seed_', 's')}={v:.0%}" if v is not None else f"{s.replace('seed_','s')}=n/a"
            for s, v in zip(seed_labels, rec_rates)
        )
        print(f"  per-seed recovery   : {per_seed_str}")

        clean = [v for v in rec_rates if v is not None]
        if len(clean) > 2:
            spread = max(clean) - min(clean)
            if spread > 0.4:
                print(f"  ⚠ HIGH VARIANCE across seeds (spread {spread:.0%}). Any claim about")
                print(f"    this fault type needs more seeds before it is trustworthy.")
        print()

        rows_out.append({
            "fault_type": fault_type,
            "n_seeds": rr_n,
            "recovery_rate_mean": round(rr_m, 4) if rr_m is not None else "",
            "recovery_rate_sd": round(rr_sd, 4) if rr_sd is not None else "",
            "degradation_rate_mean": round(dg_m, 4) if dg_m is not None else "",
            "velocity_drop_mean": round(dr_m, 4) if dr_m is not None else "",
            "settled_deficit_mean_mps": round(sd_m, 4) if sd_m is not None else "",
            "settled_deficit_sd_mps": round(sd_sd, 4) if sd_sd is not None else "",
            "fall_rate_mean": round(fr_m, 4) if fr_m is not None else "",
            "fall_rate_sd": round(fr_sd, 4) if fr_sd is not None else "",
            "recovery_time_mean_s": round(rt_m, 4) if rt_m is not None else "",
            "recovery_time_sd_s": round(rt_sd, 4) if rt_sd is not None else "",
            "post_fault_distance_mean_m": round(d_m, 4) if d_m is not None else "",
            "post_fault_distance_sd_m": round(d_sd, 4) if d_sd is not None else "",
            "per_seed_recovery_rates": "|".join(
                f"{v:.4f}" if v is not None else "" for v in rec_rates),
        })

    # headroom analysis 
    print("=" * 78)
    print("HEADROOM FOR AN ADAPTATION METHOD")
    print("=" * 78)
    ranked = sorted(rows_out, key=lambda r: r["recovery_rate_mean"]
                    if r["recovery_rate_mean"] != "" else 1.0)
    for r in ranked:
        rate = r["recovery_rate_mean"]
        if rate == "":
            continue
        if rate >= 0.95:
            note = "CEILING -- no method can improve on this; weak choice for H1"
        elif rate <= 0.30:
            note = "large headroom -- strongest test of the adaptation method"
        else:
            note = "moderate headroom"
        print(f"  {r['fault_type']:<18} baseline recovery {rate:6.1%}   {note}")

    print("\nUse this to pick the held-out fault set for the generalization")
    print("hypothesis: held-out faults should have real headroom AND be")
    print("structurally different from the training faults (physics vs. sensor).")

    # write 
    if rows_out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote across-seed summary to {args.out}")


if __name__ == "__main__":
    main()