import argparse
import csv
from collections import defaultdict
from itertools import product
from math import comb


def sign_test(diffs):
    nz = [x for x in diffs if x != 0]
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = sum(x > 0 for x in nz)
    m = max(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n
    return n, k, min(1.0, p)


def wilcoxon_exact(diffs):
    nz = [x for x in diffs if x != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    w_pos = sum(r for r, x in zip(ranks, nz) if x > 0)
    total = n * (n + 1) / 2
    w = min(w_pos, total - w_pos)
    count = sum(1 for signs in product((0, 1), repeat=n)
                if sum(r for r, s in zip(ranks, signs) if s) <= w + 1e-9)
    return min(1.0, 2 * count / 2 ** n)


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def bootstrap_ci(diffs, n_boot=10000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def load(path, exclude):
    falls = defaultdict(list)
    speeds = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["severity"]) in exclude:
                continue
            key = (r["model"], r["candidate"])
            falls[key].append(float(r["fall_rate"]))
            if r["speed"] not in ("", "nan"):
                speeds[key].append(float(r["speed"]))
    agg = {}
    for key in falls:
        agg[key] = (sum(falls[key]) / len(falls[key]),
                    sum(speeds[key]) / len(speeds[key]) if speeds[key] else None)
    return agg


def compare(agg, models, a, fx):
    pairs = [m for m in models if (m, a) in agg and (m, fx) in agg
             and agg[(m, a)][1] is not None and agg[(m, fx)][1] is not None]
    if len(pairs) < 3:
        return None
    ds = [agg[(m, fx)][1] - agg[(m, a)][1] for m in pairs]
    df = [agg[(m, fx)][0] - agg[(m, a)][0] for m in pairs]
    return {
        "n": len(pairs),
        "ds": sum(ds) / len(ds), "df": sum(df) / len(df),
        "p_s": wilcoxon_exact(ds), "p_f": wilcoxon_exact(df),
        "ci_s": bootstrap_ci(ds), "ci_f": bootstrap_ci(df),
        "faster": sum(x > 0 for x in ds), "nz_s": sum(x != 0 for x in ds),
        "safer": sum(x < 0 for x in df), "nz_f": sum(x != 0 for x in df),
    }


def run_family(agg, models, adaptive, fixed, alpha, label):
    results = [(a, fx, compare(agg, models, a, fx)) for a in adaptive for fx in fixed]
    results = [(a, fx, r) for a, fx, r in results if r is not None]
    pvals = []
    for _, _, r in results:
        pvals += [r["p_s"], r["p_f"]]
    adj = holm(pvals)
    m = len(pvals)
    floor = 2 / 2 ** min(r["n"] for _, _, r in results) if results else 1.0
    first_threshold = alpha / m if m else alpha

    print("\n" + "=" * 100)
    print(f"{label}  --  {m} tests, Holm-corrected at alpha={alpha}")
    print("=" * 100)
    if first_threshold < floor:
        print(f"  NOTE: the strictest Holm threshold ({first_threshold:.4f}) is below the "
              f"smallest achievable p ({floor:.4f}).")
        print("  At this n no single comparison in this family can reach significance;")
        print("  read the effect sizes, not the verdicts.")
    print(f"\n  {'adaptive':<16}{'vs':<12}{'speed F-A':>10}{'95% CI':>19}{'adj p':>8}"
          f"{'falls F-A':>11}{'95% CI':>19}{'adj p':>8}  verdict")
    dominated = {}
    for i, (a, fx, r) in enumerate(results):
        ap_s, ap_f = adj[2 * i], adj[2 * i + 1]
        better = ((r["ds"] > 0 and ap_s < alpha) or (r["df"] < 0 and ap_f < alpha))
        worse = ((r["ds"] < 0 and r["p_s"] < alpha) or (r["df"] > 0 and r["p_f"] < alpha))
        a_better = ((r["ds"] < 0 and ap_s < alpha) or (r["df"] > 0 and ap_f < alpha))
        a_worse = ((r["ds"] > 0 and r["p_s"] < alpha) or (r["df"] < 0 and r["p_f"] < alpha))
        if better and not worse:
            verdict = "F dominates"
            dominated.setdefault(a, []).append(fx)
        elif a_better and not a_worse:
            verdict = "A dominates"
        else:
            verdict = "-"
        ci_s = f"[{r['ci_s'][0]:+.3f},{r['ci_s'][1]:+.3f}]"
        ci_f = f"[{r['ci_f'][0]*100:+.1f},{r['ci_f'][1]*100:+.1f}]"
        print(f"  {a:<16}{fx:<12}{r['ds']:>+10.3f}{ci_s:>19}{ap_s:>8.4f}"
              f"{r['df']*100:>+9.1f}pt{ci_f:>19}{ap_f:>8.4f}  {verdict}")
    return dominated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/deployment.csv")
    parser.add_argument("--exclude_severities", type=str, default="",
                        help="Comma-separated severities to drop. A severity at which "
                             "every candidate falls ~100% adds the same amount to every "
                             "mean and compresses the differences.")
    parser.add_argument("--confirm", type=str, default=None,
                        help="Pre-specified constant(s), comma-separated, to test as the "
                             "CONFIRMATORY family. Must have been chosen before the data "
                             "being analysed, e.g. from the tuning seed.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    exclude = {float(x) for x in args.exclude_severities.split(",") if x.strip()}
    agg = load(args.csv, exclude)
    models = sorted({m for m, _ in agg})
    cands = sorted({c for _, c in agg})
    adaptive = [c for c in cands if c.startswith("adaptive")]
    fixed = [c for c in cands if c.startswith("fixed")]

    print(f"Paired analysis over {len(models)} policies"
          + (f", excluding severities {sorted(exclude)}" if exclude else ""))
    print(f"Smallest achievable exact two-sided p at n={len(models)}: "
          f"{2 / 2 ** len(models):.4f}\n")
    print(f"  {'candidate':<17}{'mean falls':>12}{'mean speed':>13}")
    for c in cands:
        fs = [agg[(m, c)][0] for m in models if (m, c) in agg]
        ss = [agg[(m, c)][1] for m in models if (m, c) in agg and agg[(m, c)][1] is not None]
        print(f"  {c:<17}{sum(fs)/len(fs):>11.1%}"
              f"{(sum(ss)/len(ss) if ss else float('nan')):>13.3f}")

    if args.confirm:
        targets = [t.strip() for t in args.confirm.split(",")]
        missing = [t for t in targets if t not in fixed]
        if missing:
            raise SystemExit(f"--confirm names not in the data: {missing}")
        dom = run_family(agg, models, adaptive, targets, args.alpha,
                         f"CONFIRMATORY: {', '.join(targets)} vs every adaptive config")
        print("\n  Confirmatory result:")
        for a in adaptive:
            print(f"    {a:<17}{'dominated' if a in dom else 'NOT dominated'}")
        print("  Both metrics are in the family. A constant that is faster but not")
        print("  detectably safer is reported as dominating only because it is also")
        print("  not detectably RISKIER; that is the claim, and no stronger one.")

    dom = run_family(agg, models, adaptive, fixed, args.alpha,
                     "EXPLORATORY: every adaptive vs every constant")
    print("\n  Non-significance means no difference was DETECTED, not that two")
    print("  conditions are equal. Report the intervals; do not call them equivalent.")


if __name__ == "__main__":
    main()
