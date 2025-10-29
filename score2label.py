import argparse
import numpy as np
import pytrec_eval
import os
import math

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--step", type=float, default=0.1)  

    args = parser.parse_args()

    with open(args.run_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    scores =[]
    for qid in run.keys():
        score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
        scores+=score_list

    scale = int(round(1 / args.step)) 
    min_scaled = math.floor(min(scores) * scale)
    max_scaled = math.ceil(max(scores) * scale)

    os.makedirs(args.output_dir, exist_ok=True)

    for s in range(min_scaled, max_scaled + 1):  
        
        t = round(s / scale, 2)
        out_path = f"{args.output_dir}/{t:.2f}.txt"
        
        with open(out_path, "w") as rj_w:
            for qid in run.keys():
                for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True):
                    if score>=t:
                        rj_w.write(f"{qid} 0 {pid} {1}\n")
                    else:
                        rj_w.write(f"{qid} 0 {pid} {0}\n")