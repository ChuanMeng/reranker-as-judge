import argparse
import os
from sys import implementation
import pytrec_eval
import json
import re
import glob

mapping = {"ndcg_cut_3": "ndcg@3",
           "ndcg_cut_5": "ndcg@5",
           "ndcg_cut_10": "ndcg@10",
           "ndcg_cut_20": "ndcg@20",
           "ndcg_cut_100": "ndcg@100",
           "ndcg_cut_1000": "ndcg@1000",
           "mrr_5": "mrr@5",
           "mrr_10": "mrr@10",
           "mrr_20": "mrr@20",
           "mrr_100": "mrr@100",
           "map_cut_10": "map@10",
           "map_cut_100": "map@100",
           "map_cut_1000": "map@1000",
           "recall_5": "recall@5",
           "recall_20": "recall@20",
           "recall_100": "recall@100",
           "recall_1000": 'recall@1000',
           "P_1": "precision@1",
           "P_3": "precision@3",
           "P_5": "precision@5",
           "P_10": "precision@10",
           "P_100": "precision@100",
           }

def evaluate_one_run(run_path: str, qrel: dict, rel_scale: int) -> dict:
    with open(run_path, 'r') as r:
        run = pytrec_eval.parse_run(r)

    #print(f"Evaluating: {run_path}")
    #print("len(list(run))", len(list(run)))
    #print("len(list(qrel))", len(list(qrel)))

    avg = {}

    # judged@K
    q2judge_10 = {}
    q2judge_20 = {}
    q2judge_100 = {}

    for qid, did_score in run.items():
        if qid not in qrel:
            continue
        sorted_did = [did for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        judge_list = [1 if docid in qrel[qid] else 0 for docid in sorted_did]
        if len(judge_list) >= 10:
            q2judge_10[qid] = sum(judge_list[0:10]) / 10
        if len(judge_list) >= 20:
            q2judge_20[qid] = sum(judge_list[0:20]) / 20
        if len(judge_list) >= 100:
            q2judge_100[qid] = sum(judge_list[0:100]) / 100

    if len(q2judge_10) > 0:
        avg[f"judge@10"] = sum(q2judge_10.values()) / len(q2judge_10)
    if len(q2judge_20) > 0:
        avg[f"judge@20"] = sum(q2judge_20.values()) / len(q2judge_20)
    if len(q2judge_100) > 0:
        avg[f"judge@100"] = sum(q2judge_100.values()) / len(q2judge_100)

    # Prepare cut runs
    run_5 = {}
    run_10 = {}
    run_20 = {}
    run_100 = {}
    for qid, did_score in run.items():
        sorted_did_score = [(did, score) for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        run_5[qid] = dict(sorted_did_score[0:5])
        run_10[qid] = dict(sorted_did_score[0:10])
        run_20[qid] = dict(sorted_did_score[0:20])
        run_100[qid] = dict(sorted_did_score[0:100])

    # Graded metrics (use original qrels)
    evaluator_ndcg = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20', 'ndcg_cut_100', 'ndcg_cut_1000'})
    results_ndcg = evaluator_ndcg.evaluate(run)

    results = {}
    for qid, _ in results_ndcg.items():
        results[qid] = {}
        for measure, score in results_ndcg[qid].items():
            results[qid][mapping[measure]] = score

    # Non-graded metrics: binarize qrels copy
    qrel_binary = {q_id: {p_id: 1 if int(rel) >= rel_scale else 0 for p_id, rel in pid_rel.items()} for q_id, pid_rel in qrel.items()}

    evaluator_general = pytrec_eval.RelevanceEvaluator(qrel_binary, {'map_cut_10', 'map_cut_100', 'map_cut_1000', 'recall_5', 'recall_20', 'recall_100', 'recall_1000', 'P_1', 'P_3', 'P_5', 'P_10', 'P_100'})
    results_general = evaluator_general.evaluate(run)

    for qid, _ in results.items():
        for measure, score in results_general[qid].items():
            results[qid][mapping[measure]] = score

    evaluator_rr = pytrec_eval.RelevanceEvaluator(qrel_binary, {'recip_rank'})
    results_rr_5 = evaluator_rr.evaluate(run_5)
    results_rr_10 = evaluator_rr.evaluate(run_10)
    results_rr_20 = evaluator_rr.evaluate(run_20)
    results_rr_100 = evaluator_rr.evaluate(run_100)

    for qid, _ in results.items():
        results[qid][mapping["mrr_5"]] = results_rr_5[qid]['recip_rank']
        results[qid][mapping["mrr_10"]] = results_rr_10[qid]['recip_rank']
        results[qid][mapping["mrr_20"]] = results_rr_20[qid]['recip_rank']
        results[qid][mapping["mrr_100"]] = results_rr_100[qid]['recip_rank']

    for measure in mapping.values():
        overall = pytrec_eval.compute_aggregated_measure(measure, [result[measure] for result in results.values()])
        avg[measure] = overall

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str) 
    parser.add_argument('--qrels_dir', type=str, required=True)
    parser.add_argument('--rel_scale', type=int, required=True)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()


    # Single run mode
    if not os.path.isdir(args.run_dir) and not os.path.isdir(args.qrels_dir):

        with open(args.qrels_dir, 'r') as r:
            qrel = pytrec_eval.parse_qrel(r)

        avg = evaluate_one_run(args.run_dir, qrel, args.rel_scale)
        
        print(avg)


        if args.output_dir is not None and args.output_dir != "":
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        
        aggregated = {}
        aggregated[os.path.basename(args.run_dir)]=avg

        
        if args.output_dir is not None and args.output_dir != "":
            data_judge=".".join(os.path.basename(args.qrels_dir).split(".")[1:-1])
            out_filename = f"ranking_performance.{data_judge}.json"
            out_path = os.path.join(args.output_dir, out_filename)

            print(data_judge)
            print(out_filename)
            print(out_path)
            #with open(out_path, 'w') as w:
            #    w.write(json.dumps(aggregated))
            #print(f"Wrote aggregated results: {out_path}")

        

    elif os.path.isdir(args.run_dir) and not os.path.isdir(args.qrels_dir):

        with open(args.qrels_dir, 'r') as r:
            qrel = pytrec_eval.parse_qrel(r)

        if args.output_dir is not None and args.output_dir != "":
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        aggregated = {}
        for fname in sorted(os.listdir(args.run_dir)):
            run_path = os.path.join(args.run_dir, fname)
            
            if not os.path.isfile(run_path):
                raise ValueError(f"Non-file entry found in runs_dir: {run_path}")

            avg = evaluate_one_run(run_path, qrel, args.rel_scale)
            aggregated[fname] = avg

        
        # Write a single JSON per dataset only if output_dir is provided
        if args.output_dir is not None and args.output_dir != "":
            data_judge=".".join(os.path.basename(args.qrels_dir).split(".")[1:-1])

            out_filename = f"ranking_performance.{data_judge}.json"
            out_path = os.path.join(args.output_dir, out_filename)
            
            #print(out_path)

            with open(out_path, 'w') as w:
                w.write(json.dumps(aggregated))
            print(f"Wrote aggregated results: {out_path}")
    
    elif os.path.isdir(args.run_dir) and os.path.isdir(args.qrels_dir):

        qrels_paths = glob.glob(os.path.join(args.qrels_dir, "*"))

        data_judge=".".join(os.path.basename(args.qrels_dir).split(".")[1:]) # be careful here, should be [1:]
        sub_dir = os.path.join(args.output_dir, f"ranking_performance.{data_judge}")

        os.makedirs(sub_dir, exist_ok=True)

        for qrels_path in qrels_paths:
            indicator = float(".".join(qrels_path.split("/")[-1].split(".")[:-1]))
            
            with open(qrels_path, 'r') as r:
                qrel = pytrec_eval.parse_qrel(r)

            aggregated = {}
            for fname in sorted(os.listdir(args.run_dir)):
                run_path = os.path.join(args.run_dir, fname)
            
                if not os.path.isfile(run_path):
                    raise ValueError(f"Non-file entry found in runs_dir: {run_path}")

                avg = evaluate_one_run(run_path, qrel, args.rel_scale)
                aggregated[fname] = avg
        
            # Write a single JSON per dataset only if output_dir is provided
            if args.output_dir:
                
                out_path = os.path.join(sub_dir, f"{indicator}.json")
                with open(out_path, 'w') as w:
                    w.write(json.dumps(aggregated))
                print(f"Wrote aggregated results: {out_path}")
        
    else:
        raise Exception
