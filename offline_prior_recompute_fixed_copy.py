#!/usr/bin/env python3
import argparse, json, math, statistics
from collections import defaultdict, Counter
from pathlib import Path

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def mean(xs):
    return statistics.fmean(xs) if xs else 0.0

def normalize(vals):
    vals = [v if math.isfinite(v) and v >= 0 else 0.0 for v in vals]
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / s for v in vals]

def entropy(ps):
    return -sum(p * math.log(p + 1e-12) for p in ps)

def top_gap(ps):
    ss = sorted(ps, reverse=True)
    return ss[0] - ss[1] if len(ss) >= 2 else 0.0

def argmax(xs):
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs[1:], 1):
        if v > best_v:
            best_i, best_v = i, v
    return best_i

def get_correct(row):
    # support multiple possible field names
    for k in ["answer_correctness", "answer_correct", "is_correct", "correct"]:
        if k in row:
            return 1.0 if safe_float(row.get(k), 0.0) > 0.5 else 0.0
    return 0.0

def get_likelihood(row):
    for k in ["likelihood", "evidence_likelihood"]:
        if k in row:
            return safe_float(row.get(k), 0.0)
    return 0.0

def get_prior(row):
    for k in ["prior_probability", "prior", "stored_prior"]:
        if k in row:
            return safe_float(row.get(k), 1.0)
    return 1.0

def get_error_type(row):
    if isinstance(row.get("error_type"), str):
        return row["error_type"]
    ev = row.get("evidence")
    if isinstance(ev, dict) and isinstance(ev.get("error_type"), str):
        return ev["error_type"]
    return "unknown"

def problem_id(row):
    return str(row.get("problem_id", row.get("unique_id", "unknown")))

def rollout_sort_key(row):
    v = row.get("rollout_id", 0)
    try:
        return float(v)
    except Exception:
        return str(v)

def analyze(groups, lam, high_threshold=0.2):
    posterior_top1 = []
    likelihood_top1 = []
    outcome_top1 = []
    prior_only_top1 = []
    oracle = []
    entropies = []
    gaps = []
    mass_correct = []
    unique_adv = []

    like_correct, like_incorrect = [], []
    post_correct, post_incorrect = [], []

    incorrect_high = 0
    wrong_high = 0
    incorrect_count = 0
    total = 0

    for rows in groups.values():
        rows = sorted(rows, key=rollout_sort_key)
        n = len(rows)
        if n == 0:
            continue

        corrects = [get_correct(r) for r in rows]
        likes = [get_likelihood(r) for r in rows]

        raw_priors = normalize([get_prior(r) for r in rows])

        if lam == 0.0:
            effective_priors = [1.0 / n] * n
        else:
            effective_priors = normalize([p ** lam for p in raw_priors])

        weighted = [p * l for p, l in zip(effective_priors, likes)]
        post = normalize(weighted)

        pi = argmax(post)
        li = argmax(likes)
        oi = argmax(corrects)
        pri = argmax(effective_priors)

        posterior_top1.append(corrects[pi])
        likelihood_top1.append(corrects[li])
        outcome_top1.append(1.0 if max(corrects) > 0 else 0.0)
        prior_only_top1.append(corrects[pri])
        oracle.append(1.0 if max(corrects) > 0 else 0.0)

        entropies.append(entropy(post))
        gaps.append(top_gap(post))
        mass_correct.append(sum(p for p, c in zip(post, corrects) if c > 0.5))

        m = mean(post)
        unique_adv.append(len(set(round(p - m, 12) for p in post)))

        for r, c, l, p in zip(rows, corrects, likes, post):
            total += 1
            if c > 0.5:
                like_correct.append(l)
                post_correct.append(p)
            else:
                incorrect_count += 1
                like_incorrect.append(l)
                post_incorrect.append(p)
                if p >= high_threshold:
                    incorrect_high += 1
                    if get_error_type(r) == "wrong_direction":
                        wrong_high += 1

    return {
        "prior_lambda": lam,
        "num_problems": len(groups),
        "num_rollouts_total": total,
        "oracle_best_of_n_accuracy": mean(oracle),
        "outcome_top1_accuracy": mean(outcome_top1),
        "likelihood_top1_accuracy": mean(likelihood_top1),
        "posterior_top1_accuracy": mean(posterior_top1),
        "prior_only_top1_accuracy": mean(prior_only_top1),
        "posterior_mass_on_correct_mean": mean(mass_correct),
        "posterior_entropy_mean": mean(entropies),
        "posterior_top1_top2_gap_mean": mean(gaps),
        "posterior_advantage_unique_values_per_problem_mean": mean(unique_adv),
        "incorrect_but_high_posterior_count": incorrect_high,
        "incorrect_but_high_posterior_rate": incorrect_high / incorrect_count if incorrect_count else 0.0,
        "wrong_direction_high_posterior_count": wrong_high,
        "avg_likelihood_correct": mean(like_correct),
        "avg_likelihood_incorrect": mean(like_incorrect),
        "likelihood_correct_incorrect_gap": mean(like_correct) - mean(like_incorrect),
        "avg_posterior_correct": mean(post_correct),
        "avg_posterior_incorrect": mean(post_incorrect),
        "posterior_correct_incorrect_gap": mean(post_correct) - mean(post_incorrect),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--high_posterior_threshold", type=float, default=0.2)
    args = ap.parse_args()

    rows = []
    diagnostics = {
        "num_json_lines": 0,
        "answer_correctness_value_counts": Counter(),
        "answer_correctness_source_counts": Counter(),
        "likelihood_source_counts": Counter(),
        "prior_source_counts": Counter(),
    }

    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            diagnostics["num_json_lines"] += 1

            # source diagnostics
            for k in ["answer_correctness", "answer_correct", "is_correct", "correct"]:
                if k in r:
                    diagnostics["answer_correctness_source_counts"][k] += 1
                    break
            diagnostics["answer_correctness_value_counts"][str(get_correct(r))] += 1

            for k in ["likelihood", "evidence_likelihood"]:
                if k in r:
                    diagnostics["likelihood_source_counts"][k] += 1
                    break

            for k in ["prior_probability", "prior", "stored_prior"]:
                if k in r:
                    diagnostics["prior_source_counts"][k] += 1
                    break

    groups = defaultdict(list)
    for r in rows:
        groups[problem_id(r)].append(r)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for lam in args.lambdas:
        tag = str(lam).replace(".", "p")
        metrics = analyze(groups, lam, args.high_posterior_threshold)
        payload = {
            "comparison_note": "Same fixed rollout pool. Strategy prior is exponentiated by prior_lambda; lambda=0 is uniform/likelihood-only.",
            "input_path": args.input_path,
            "load_diagnostics": {
                k: dict(v) if isinstance(v, Counter) else v
                for k, v in diagnostics.items()
            },
            "metrics": metrics,
        }
        out_path = out_dir / f"offline_prior_lambda_{tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        all_metrics[str(lam)] = metrics
        print(f"saved {out_path}")

    summary_path = out_dir / "offline_prior_lambda_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "input_path": args.input_path,
            "load_diagnostics": {
                k: dict(v) if isinstance(v, Counter) else v
                for k, v in diagnostics.items()
            },
            "lambda_results": all_metrics,
        }, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print("\nSummary:")
    print("lambda | top1 | mass_correct | entropy | gap | adv_unique | incorrect_high | wrong_high")
    for lam in args.lambdas:
        m = all_metrics[str(lam)]
        print(
            f"{lam:>5} | "
            f"{m['posterior_top1_accuracy']:.4f} | "
            f"{m['posterior_mass_on_correct_mean']:.4f} | "
            f"{m['posterior_entropy_mean']:.4f} | "
            f"{m['posterior_top1_top2_gap_mean']:.4f} | "
            f"{m['posterior_advantage_unique_values_per_problem_mean']:.2f} | "
            f"{m['incorrect_but_high_posterior_count']} | "
            f"{m['wrong_direction_high_posterior_count']}"
        )
    print(f"\nsaved summary {summary_path}")

if __name__ == "__main__":
    main()