import argparse
import json
import os
from typing import Dict, List
import glob
from scipy.stats import kendalltau, spearmanr


def load_ranking_performance(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as r:
        data = json.load(r)
    # Expecting: { system_name: { metric: value, ... }, ... }
    return data

def extract_metric_vectors(
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    metric: str,
) -> List[List[float]]:

    systems = list(a.keys())
    xs, ys = [], []

    for sys in systems:
        xs.append(float(a[sys][metric]))
        ys.append(float(b[sys][metric]))

    return xs, ys, systems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_dir", type=str, required=True, help="should be human")
    parser.add_argument("--b_dir", type=str, required=True, help="should be a system")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Metrics to evaluate, e.g., ndcg@10 mrr@10 map@100",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional path to write correlations as JSON (e.g., /path/to/indicate.json)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.b_dir):

        results = {}

        perf_a = load_ranking_performance(args.a_dir)
        perf_b = load_ranking_performance(args.b_dir)

        print(f"Loaded systems: A={len(perf_a)}, B={len(perf_b)}")

        systems_a = set(perf_a.keys())
        systems_b = set(perf_b.keys())
        if systems_a != systems_b:
            only_in_a = sorted(systems_a - systems_b)
            only_in_b = sorted(systems_b - systems_a)
            raise ValueError(
                "System keys mismatch between files. "
                f"Only in A ({len(only_in_a)}): {only_in_a[:10]}{' ...' if len(only_in_a) > 10 else ''}; "
                f"Only in B ({len(only_in_b)}): {only_in_b[:10]}{' ...' if len(only_in_b) > 10 else ''}"
            )

        total_systems = len(systems_a)
        if total_systems < 2:
            raise ValueError(f"Insufficient systems to compute correlations (n={total_systems}); need at least 2")

    
        result = {
        "n_systems": total_systems,
        "files": {"A": os.path.abspath(args.a_dir), "B": os.path.abspath(args.b_dir)},
        "metrics": {},

    }   

        for metric in args.metrics:
            xs, ys, systems = extract_metric_vectors(perf_a, perf_b, metric)

            ktau, ktau_p = kendalltau(xs, ys)
            spr, spr_p = spearmanr(xs, ys)

            result["metrics"][metric] = {
            "kendall_tau": ktau,
            "kendall_tau_p": ktau_p,
            "spearman_rho": spr,
            "spearman_rho_p": spr_p,
            #"systems_used": systems,  
        }

            print(f"Metric: {metric}")
            print(f"  systems_used: {total_systems}")
            print(f"  kendall_tau: {ktau:.3f}  p_value: {ktau_p:.6g}")
            print(f"  spearman_rho: {spr:.3f}  p_value: {spr_p:.6g}")
        

        results["gen"]=result

    else:
        perf_a = load_ranking_performance(args.a_dir)

        paths = glob.glob(os.path.join(args.b_dir, "*"))

        results={}

        for path in paths:
            indicator = float(".".join(path.split("/")[-1].split(".")[:-1]))
            
            perf_b = load_ranking_performance(path)

            print(f"Loaded systems: A={len(perf_a)}, B={len(perf_b)}")

            systems_a = set(perf_a.keys())
            systems_b = set(perf_b.keys())
            if systems_a != systems_b:
                only_in_a = sorted(systems_a - systems_b)
                only_in_b = sorted(systems_b - systems_a)
                raise ValueError(
                    "System keys mismatch between files. "
                    f"Only in A ({len(only_in_a)}): {only_in_a[:10]}{' ...' if len(only_in_a) > 10 else ''}; "
                    f"Only in B ({len(only_in_b)}): {only_in_b[:10]}{' ...' if len(only_in_b) > 10 else ''}"
                )

            total_systems = len(systems_a)
            if total_systems < 2:
                raise ValueError(f"Insufficient systems to compute correlations (n={total_systems}); need at least 2")
    
            result = {
            "n_systems": total_systems,
            "files": {"A": os.path.abspath(args.a_dir), "B": os.path.abspath(path)},
            "metrics": {},
            }   

            for metric in args.metrics:
                xs, ys, systems = extract_metric_vectors(perf_a, perf_b, metric)

                ktau, ktau_p = kendalltau(xs, ys)
                spr, spr_p = spearmanr(xs, ys)

                result["metrics"][metric] = {
                "kendall_tau": ktau,
                "kendall_tau_p": ktau_p,
                "spearman_rho": spr,
                "spearman_rho_p": spr_p,
                #"systems_used": systems,  
                }

            results[indicator] = result
    

    if args.output_path is not None:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"result has been saved to: {args.output_path}")
            

if __name__ == "__main__":
    main()


