import argparse, json, csv
from pathlib import Path
import math

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def best_threshold_system(data, metric):
    best_t, best_v = None, float("-inf")
    for t, rec in data.items():
        try:
            v = rec["metrics"][metric]["kendall_tau"]
            if isinstance(v, (int, float)) and v > best_v:
                best_t, best_v = t, v
        except Exception:
            pass
    return best_t, best_v

def best_threshold_rj(data):
    best_t, best_v = None, float("-inf")
    for t, rec in data.items():
        v = rec.get("cohen_kappa")
        if isinstance(v, (int, float)) and v > best_v:
            best_t, best_v = t, v
    return best_t, best_v

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_json", required=True)
    ap.add_argument("--tgt_json", required=True)
    ap.add_argument("--task", choices=["system_ranking", "rj"], required=True)
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    src = load_json(args.src_json)
    tgt = load_json(args.tgt_json)

    if args.task == "system_ranking":
        best_t, src_score = best_threshold_system(src, args.metric)
        tgt_score = tgt.get(best_t, {}).get("metrics", {}).get(args.metric, {}).get("kendall_tau", None)
        metric_label = f"{args.metric} (Kendall τ)"
    else:
        best_t, src_score = best_threshold_rj(src)
        tgt_score = tgt.get(best_t, {}).get("cohen_kappa", None)
        metric_label = "Cohen’s κ"

    # 格式化数值
    src_score_str = fmt(src_score)
    tgt_score_str = fmt(tgt_score)

    file_exists = Path(args.csv).exists()
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["direction", "judge", "task", "metric", "src_json", "tgt_json", "src_threshold", "src_score", "tgt_score"])
        writer.writerow([args.direction, args.judge, args.task, metric_label, args.src_json, args.tgt_json, best_t, src_score_str, tgt_score_str])

if __name__ == "__main__":
    main()
