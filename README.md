# Reproducing Re-Rankers as Relevance Judges

This is the repository for the paper titled **Reproducing Re-Rankers as Relevance Judges**.
In this paper, we reproduce three re-rankers from different re-ranker families (monoT5, RankLLaMA, and Rank1) as relevance judgment predictors (a.k.a. relevance judges).

The code in this repository can reproduce all results reported in the paper.

This repository is structured into five distinct parts:
- [1. Prerequest](#1-prerequest)
- [2. Data preparation](#2-data-preparation)
- [3. Reproducing results](#3-reproducing-results)
  - [3.1 RQ1: Re-rankers as judges via direct generation](#31-rq1-re-rankers-as-judges-via-direct-generation)
  - [3.2 RQ2: Re-rankers as judges via score thresholding](#32-rq2-re-rankers-as-judges-via-score-thresholding)
  - [3.3 RQ3: Bias of re-ranker-based judges towards re-rankers](#33-rq3-bias-of-re-ranker-based-judges-towards-re-rankers)
- [4. Create plots](#4-create-plots)
- [5. Results on nDCG@10 (We report results on ndcg10 here because of limited space in the paper)](#5-results-on-ndcg10-we-report-results-on-ndcg10-here-because-of-limited-space-in-the-paper)

## 1. Prerequest
We recommend executing all processes in a Linux environment.
Install dependencies:
```bash
pip install -r requirements.txt
```
Also, please install [Tevatron](https://github.com/texttron/tevatron) and [Pyserini](https://github.com/castorini/pyserini) in advance.

## 2. Data preparation
### 2.1 Create folders
Create the following folders:
```bash
mkdir -p \
data/{queries,qrels,indexes,corpus,runs} \
output/{rq1_2/{rerank_input,qrels,runs},rq3/{rerank_input,qrels,runs}} \
result/{rq1_2/{rj,system_ranking},rq3/system_ranking} \
plots/{rq1_2,rq3}
```
### 2.2 Download queries, qrels, and collections
Download queries for TREC-DL 2019 to 2023:
```bash
wget -P  data/queries/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl19-passage.txt
wget -P  data/queries/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl20.txt
wget -P  data/queries/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl21.txt
wget -P  data/queries/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl22.txt
wget -P  data/queries/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/topics.dl23.txt
```

Download qrels files for TREC-DL 2019 to 2023:
```bash
wget -P  data/qrels/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl19-passage.txt
wget -P  data/qrels/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl20-passage.txt
wget -P  data/qrels/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl21-passage.txt
wget -P  data/qrels/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl22-passage.txt
wget -P  data/qrels/ https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/qrels.dl23-passage.txt

mv data/qrels/qrels.dl19-passage.txt data/qrels/qrels.nist.dl19-passage.txt
mv data/qrels/qrels.dl20-passage.txt data/qrels/qrels.nist.dl20-passage.txt
mv data/qrels/qrels.dl21-passage.txt data/qrels/qrels.nist.dl21-passage.txt
mv data/qrels/qrels.dl22-passage.txt data/qrels/qrels.nist.dl22-passage.txt
mv data/qrels/qrels.dl23-passage.txt data/qrels/qrels.nist.dl23-passage.txt
```

Download and process corpus files for MS MARCO V1 and V2 passage ranking collections:
```bash
# MS MARCO V1
wget -P data/indexes/ https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz --no-check-certificate
tar -zxvf  data/indexes/lucene-index.msmarco-v1-passage-full.20221004.252b5e.tar.gz -C data/indexes/

wget -P ./data/corpus/ https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz --no-check-certificate
tar -zxvf  ./data/corpus/collection.tar.gz  -C ./data/corpus/
mv ./data/corpus/collection.tsv ./data/corpus/msmarco_v1_passage.tsv

# MS MARCO V2
wget -P data/indexes https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz --no-check-certificate
tar -zxvf  data/indexes/lucene-index.msmarco-v2-passage-full.20220808.4d6d2a.tar.gz -C data/indexes/

wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar -P ./data/corpus/
tar -xvf ./data/corpus/msmarco_v2_passage.tar -C ./data/corpus/

for file in ./data/corpus/msmarco_v2_passage/*.gz; do
    gzip -d "$file"
done
```

### 2.3 Download submitted run files
Please download the submitted run files for TREC-DL 2019, 2020, 2021, 2022, and 2023, and place them in the corresponding folders: `./data/runs/runs.dl19-passage`, `./data/runs/runs.dl20-passage`, `./data/runs/runs.dl21-passage`, .`/data/runs/runs.dl22-passage`, and `./data/runs/runs.dl23-passage`.
Please make sure that every run file is unzipped and its compressed version deleted.

### 2.4 Request depubliced qrels and run files
Following the paper [UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor](https://arxiv.org/pdf/2406.06519), we use the depubliced versions of the qrels and submitted run files for TREC-DL 2022 and 2023.
Please contact Shivani Upadhyay, the first author of the paper [UMBRELA](https://arxiv.org/pdf/2406.06519), to request access to these depubliced qrels (`qrels.dl23-passage_duped` and `qrels.dl23-passage_duped`) and run files (`runs.dl23-passage_duped` and `runs.dl23-passage_duped`).

Please place the depubliced qrels (`qrels.dl22-passage_duped`, `qrels.dl23-passage_duped`) in the folder ./data/qrels.
And please rename them:
```bash
mv data/qrels/qrels.dl22-passage-duped.txt data/qrels/qrels.nist.dl22-passage_duped.txt
mv data/qrels/qrels.dl23-passage-duped.txt data/qrels/qrels.nist.dl23-passage_duped.txt
```

Also, please place the depubliced runs (`runs.dl22-passage_duped`, `runs.dl23-passage_duped`) in the folder ./data/runs.

### 2.5 Request UMBRELA's predicted relevance judgments
Please contact Shivani Upadhyay, the first author of the paper [UMBRELA](https://arxiv.org/pdf/2406.06519), to request access to UMBRELA's predicted relevance judgments for TREC-DL 2019 to 2023.
Place these files in the folder `./output/rq1_2/qrels/`.
And please rename these files as `qrels.gpt-4o_0123_100_0_1.dl19-passage.txt`, `qrels.gpt-4o_0123_100_0_1.dl20-passage.txt`, `qrels.gpt-4o_0123_100_0_1.dl21-passage.txt`, `qrels.gpt-4o_0123_100_0_1.dl22-passage_duped.txt` and `qrels.gpt-4o_0123_100_0_1.dl23-passage_duped.txt`.

## 3. Reproducing results

### 3.1 RQ1: Re-rankers as judges via direct generation

#### 3.1.1 Generate re-ranker input files
Run the following commands to generate re-ranker input files for [monoT5](https://aclanthology.org/2020.findings-emnlp.63/), [RankLLaMA](https://dl.acm.org/doi/10.1145/3626772.3657951), and [Rank1](https://openreview.net/pdf?id=Pg0PAvbhGv) (We use a unified input format for all re-rankers):
```bash
for d in 19 20 21 22 23
do
    if [[ $d -eq 19 || $d -eq 20 ]]; then
        corpus_name="msmarco_v1_passage.tsv"
    else
        corpus_name="msmarco_v2_passage"
    fi

    if [[ $d -eq 19 ]]; then
        query_name="topics.dl${d}-passage.txt"
    else
        query_name="topics.dl${d}.txt"
    fi

    if [[ $d -eq 22 || $d -eq 23 ]]; then
        qrels_name="qrels.nist.dl${d}-passage_duped.txt"
        output_name="rerank_input.dl${d}-passage_duped.jsonl"
    else
        qrels_name="qrels.nist.dl${d}-passage.txt"
        output_name="rerank_input.dl${d}-passage.jsonl"
    fi

    echo dl${d}, ${qrels_name}, ${corpus_name}, ${query_name},${output_name}
    python prepare_rerank_file.py \
        --qrels_path ./data/qrels/${qrels_name} \
        --corpus_path ./data/corpus/${corpus_name} \
        --query_path ./data/queries/${query_name} \
        --output_path ./output/rq1_2/rerank_input/${output_name}
done
```
The generated re-ranker input files will be saved in `./output/rq1_2/rerank_input/`.

#### 3.1.2 Run monoT5 and Rank1
Run the following commands to execute monoT5 (base, large, and 3B) and Rank1 (7B, 14B, and 32B) to directly generate predicted relevance judgments (qrels) for the query–document pairs in the original TREC-DL 2019–2023 qrels:
```bash
# monoT5
GPU_ID=0
BATCH_SIZE=8

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for m in "castorini/monot5-base-msmarco" "castorini/monot5-large-msmarco" "castorini/monot5-3b-msmarco"
do
for d in 19 20 21 22 23
do
   echo ">>> Starting model ${m} on dataset dl${d} at $(date)"
   if [[ $d -eq 22 || $d -eq 23 ]]; then
        dataset_name="rerank_input.dl${d}-passage_duped.jsonl"
        rerank_output_name="run.${m##*/}.dl${d}-passage_duped.txt"
        qrels_output_name="qrels.${m##*/}-gen.dl${d}-passage_duped.txt"

    else
        dataset_name="rerank_input.dl${d}-passage.jsonl"
        rerank_output_name="run.${m##*/}.dl${d}-passage.txt"
        qrels_output_name="qrels.${m##*/}-gen.dl${d}-passage.txt"
    fi

python -u monot5.py \
--model_name_or_path "${m}" \
--tokenizer_name "${m}" \
--dataset_path ./output/rq1_2/rerank_input/${dataset_name} \
--rerank_output_path ./output/rq1_2/runs/${rerank_output_name} \
--qrels_output_path ./output/rq1_2/qrels/${qrels_output_name} \
--batch_size 8
echo ">>> Finish model ${m} on dataset dl${d} at $(date)"
done
done

# Rank1
GPU_ID=0
BATCH_SIZE=512

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for m in "jhu-clsp/rank1-7b" "jhu-clsp/rank1-14b" "jhu-clsp/rank1-32b-awq"
do
for d in 19 20 21 22 23
do
   echo ">>> Starting model ${m} on dataset dl${d} at $(date)"
   if [[ $d -eq 22 || $d -eq 23 ]]; then
        dataset_name="rerank_input.dl${d}-passage_duped.jsonl"
        rerank_output_name="run.${m##*/}.dl${d}-passage_duped.txt"
        qrels_output_name="qrels.${m##*/}-gen.dl${d}-passage_duped.txt"

    else
        dataset_name="rerank_input.dl${d}-passage.jsonl"
        rerank_output_name="run.${m##*/}.dl${d}-passage.txt"
        qrels_output_name="qrels.${m##*/}-gen.dl${d}-passage.txt"
    fi

python -u rank1.py \
--model_name_or_path "${m}" \
--dataset_path ./output/rq1_2/rerank_input/${dataset_name} \
--rerank_output_path ./output/rq1_2/runs/${rerank_output_name} \
--qrels_output_path ./output/rq1_2/qrels/${qrels_output_name} \
--batch_size $BATCH_SIZE
echo ">>> Finish model ${m} on dataset dl${d} at $(date)"
done
done
```
Because monoT5 and Rank1 can both directly generate relevance judgments and re-ranking scores, the above commands produce predicted relevance judgments for the same query–document pairs as in the original TREC-DL qrels (saved in `./output/rq1_2/qrels/`) and re-ranking scores for all query–document pairs (saved in `./output/rq1_2/runs/`).
Please note that re-ranking scores will be used in 3.2.

#### 3.1.3 Measure the ranking performance of the submitted runs

We first measure the ranking performance of the submitted runs using human-based relevance judgments for reference:
```bash
for judge in nist
do
for d in 19 20 21 22 23
do
 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./data/runs/runs.${judgeset} \
--qrels_dir ./data/qrels/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq1_2/ranking_performance \
--rel_scale 2
done
done
```
Files with ranking performance saved in `./output/rq1_2/ranking_performance`.

We also measure the ranking performance of the submitted runs using both human and UMBRELLA-based relevance judgments for reference:
```bash
for judge in nist gpt-4o_0123_100_0_1
do
for d in 19 20 21 22 23
do
 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./data/runs/runs.${judgeset} \
--qrels_dir ./output/rq1_2/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq1_2/ranking_performance \
--rel_scale 2
done
done
```

We then measure the ranking performance of the submitted runs using relevance judgment generated by the re-rankers:
```bash
for judge in monot5-base-gen monot5-large-gen monot5-3b-gen rank1-7b-gen rank1-14b-gen rank1-32b-awq-gen
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi
echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./data/runs/runs.${judgeset} \
--qrels_dir ./output/rq1_2/qrels/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq1_2/ranking_performance \
--rel_scale 1
done
```

### 3.1.3 System ranking evaluation
We first evaluate the correlation between the system rankings produced by UMBRELLA and those produced by human judges for reference:
```bash
judge=gpt-4o_0123_100_0_1
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_system_ranking.py \
--a_dir ./output/rq1_2/ranking_performance/ranking_performance.nist.${judgeset}.json \
--b_dir ./output/rq1_2/ranking_performance/ranking_performance.${judge}.${judgeset}.json \
--output_path ./result/rq1_2/system_ranking/result.${judge}.${judgeset}.json \
--metrics ndcg@10 map@100 mrr@10
done
```
The result files saved in `./result/rq1_2/system_ranking/`.

We then evaluate the correlation between the system rankings produced by the re-ranker-based judges and those produced by each re-ranker-based relevance judge:
```bash

for judge in monot5-base-gen monot5-large-gen monot5-3b-gen rank1-7b-gen rank1-14b-gen rank1-32b-awq-gen
do
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_system_ranking.py \
--a_dir ./output/rq1_2/ranking_performance/ranking_performance.nist.${judgeset}.json \
--b_dir ./output/rq1_2/ranking_performance/ranking_performance.${judge}.${judgeset}.json \
--output_path ./result/rq1_2/system_ranking/result.${judge}.${judgeset}.json \
--metrics ndcg@10 map@100 mrr@10
done
done
```

### 3.1.4 Relevance judgment aggrement evaluation
We first measure the relevance judgment agreement between UMBRELLA and human annotators for reference.
```bash
judge=gpt-4o_0123_100_0_1
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_rj.py \
--qrels_true_path ./data/qrels/qrels.nist.${judgeset}.txt \
--qrels_pred_path ./output/rq1_2/qrels/qrels.${judge}.${judgeset}.txt \
--output_path ./result/rq1_2/rj/result.${judge}.${judgeset}.json \
--binary
done
```
The result files saved in `./result/rq1_2/rj/`.

We then measure the relevance judgment agreement (Cohen's kappa) between each re-ranker-based judge and human annotators:
```bash
for judge in monot5-base-gen monot5-large-gen monot5-3b-gen rank1-7b-gen rank1-14b-gen rank1-32b-awq-gen
do
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_rj.py \
--qrels_true_path ./data/qrels/qrels.nist.${judgeset}.txt \
--qrels_pred_path ./output/rq1_2/qrels/qrels.${judge}.${judgeset}.txt \
--output_path ./result/rq1_2/rj/result.${judge}.${judgeset}.json \
--binary --pre_is_binary
done
done
```

### 3.2 RQ2: Re-rankers as judges via score thresholding

#### 3.2.1 Run RankLLaMA
Because we already obtained the re-ranking scores for the query–document pairs in the original TREC-DL 2019–2023 qrels in 3.1.2, we only need to generate RankLLaMA’s outputs here.
We use the re-ranker input files produced in 3.1.1.
Run the following commands to execute RankLLaMA and generate its re-ranking scores for the query–document pairs in the original TREC-DL 2019–2023 qrels:
```bash
GPU_ID=0
BATCH_SIZE=32

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

tokenizers=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
)

models=(
  "castorini/rankllama-v1-7b-lora-passage"
  "castorini/rankllama-v1-13b-lora-passage"
)

for i in "${!models[@]}"; do
  t="${tokenizers[$i]}"
  m="${models[$i]}"

  for d in 19 20 21 22 23; do
    echo ">>> Starting model ${m} on dataset dl${d} at $(date)"

    if [[ $d -eq 22 || $d -eq 23 ]]; then
      dataset_name="rerank_input.dl${d}-passage_duped.jsonl"
      rerank_output_name="run.${m##*/}.dl${d}-passage_duped.txt"
      tmp_name="tmp_run.${m##*/}.dl${d}-passage_duped.txt"
    else
      dataset_name="rerank_input.dl${d}-passage.jsonl"
      rerank_output_name="run.${m##*/}.dl${d}-passage.txt"
      tmp_name="tmp_run.${m##*/}.dl${d}-passage.txt"
    fi

    python -m tevatron.reranker.driver.rerank \
      --output_dir=temp \
      --model_name_or_path "$m" \
      --tokenizer_name "$t" \
      --dataset_path "./output/rq1_2/rerank_input/${dataset_name}" \
      --fp16 \
      --per_device_eval_batch_size "${BATCH_SIZE}" \
      --rerank_max_len $((32 + 164)) \
      --dataset_name json \
      --query_prefix "query: " \
      --passage_prefix "document: " \
      --rerank_output_path "./output/rq1_2/runs/${tmp_name}"

    python -m tevatron.utils.format.convert_result_to_trec \
      --input "./output/rq1_2/runs/${tmp_name}" \
      --output "./output/rq1_2/runs/${rerank_output_name}"

    rm -f "./output/rq1_2/runs/${tmp_name}"

    echo ">>> Finished model ${m} on dataset dl${d} at $(date)"
  done
done
```
The produced re-ranking scores saved in `./output/rq1_rq2/runs/`.

#### 3.2.2 Generate qrels at different thresholds

The following commands generate monoT5's qrels at each threshold (with a step size of 0.01):
```bash
for m monot5-base monot5-large monot5-3b
do
for d in 19 20 21 22 23
do

if [[ $d -eq 22 || $d -eq 23 ]]; then
        run_name="run.${m}.dl${d}-passage_duped.txt"
        output_name="qrels.${m}.dl${d}-passage_duped"
    else
        run_name="run.${m}.dl${d}-passage.txt"
        output_name="qrels.${m}.dl${d}-passage"
    fi

python -u score2label.py \
--run_path ./output/rq1_2/runs/${run_name} \
--output_dir ./output/rq1_2/qrels/${output_name} \
--step 0.01
done
done
```
The resulting folders, each containing qrels at different thresholds, will be saved under `./output/rq1_2/qrels/`.

Similaly, the following commands generate Rank1's qrels at each threshold (with a step size of 0.01):
```bash
for m rank1-7b rank1-14b rank1-32b-awq
do
for d in 19 20 21 22 23
do

if [[ $d -eq 22 || $d -eq 23 ]]; then
        run_name="run.${m}.dl${d}-passage_duped.txt"
        output_name="qrels.${m}.dl${d}-passage_duped"
    else
        run_name="run.${m}.dl${d}-passage.txt"
        output_name="qrels.${m}.dl${d}-passage"
    fi

python -u score2label.py \
--run_path ./output/rq1_2/runs/${run_name} \
--output_dir ./output/rq1_2/qrels/${output_name} \
--step 0.01
done
done
```
Slightly differently, the following commands generate RankLLaMA’s predicted qrels at each threshold (with a step size of 0.1, since RankLLaMA produces a much wider range of re-ranking scores):
```bash
for m rankllama-7b rankllama-13b
do
for d in 19 20 21 22 23
do

if [[ $d -eq 22 || $d -eq 23 ]]; then
        run_name="run.${m}.dl${d}-passage_duped.txt"
        output_name="qrels.${m}.dl${d}-passage_duped"
    else
        run_name="run.${m}.dl${d}-passage.txt"
        output_name="qrels.${m}.dl${d}-passage"
    fi

python -u score2label.py \
--run_path ./output/rq1_2/runs/${run_name} \
--output_dir ./output/rq1_2/qrels/${output_name} \
--step 0.1
done
done
```

#### 3.1.3 Measure the ranking performance of the submitted runs at different thresholds

We measure the ranking performance of the submitted runs using re-ranker-based judge at each threshold:
```bash
for judge in monot5-base monot5-large monot5-3b rankllama-7b rankllama-13b rank1-7b rank1-14b rank1-32b-awq
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./data/runs/runs.${judgeset} \
--qrels_dir ./output/rq1_2/qrels/qrels.${judge}.${judgeset} \ # contain qrels at different thresholds
--output_dir ./output/rq1_2/ranking_performance \
--rel_scale 1
done
```
The above commands will generate a subfolder for each judge on each dataset under `./output/rq1_2/ranking_performance`.
Each subfolder contains a set of files, where each file corresponds to the ranking performance evaluated at a particular threshold.

#### 3.2.4 System ranking evaluation at different thresholds

We evaluate the correlation between the system rankings produced by the re-ranker-based judges and those produced by each re-ranker-based relevance judge at different thresholds:
```bash
for judge in monot5-base monot5-large monot5-3b rankllama-7b rankllama-13b rank1-7b rank1-14b rank1-32b-awq
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo -----${d} -----
python -u evaluate_system_ranking.py \
--a_dir ./output/rq1_2/ranking_performance/ranking_performance.nist.${judgeset}.json \
--b_dir ./output/rq1_2/ranking_performance/ranking_performance.${judge}.${judgeset} \ # a folder, not a file
--output_path ./result/rq1_2/system_ranking/result.${judge}.${judgeset}.json \
--metrics ndcg@10 map@100 mrr@10
done
done
```
The above commands will produce one result file (ending with .json) for each judge on each dataset under `./result/rq1_2/system_ranking/`.
Each result file contains correlation values across different thresholds.

### 3.2.5 Relevance judgment aggrement evaluation at different thresholds

We then measure the relevance judgment agreement (Cohen's kappa) between human annotators and each re-ranker-based judge at different thresholds:
```bash
for judge in monot5-base monot5-large monot5-3b rankllama-7b rankllama-13b rank1-7b rank1-14b rank1-32b-awq
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_rj.py \
--qrels_true_path ./data/qrels/qrels.nist.${judgeset}.txt \
--qrels_pred_path ./output/rq1_2/qrels/qrels.${judge}.${judgeset} \ # a folder, not a file
--output_path ./result/rq1_2/rj/result.${judge}.${judgeset}.json \
--binary --pre_is_binary
done
done
```
The above commands will produce one result file (ending with .json) for each judge on each dataset under `./result/rq1_2/rj/`.
Each result file contains the agreement values (Cohen’s κ) computed across different thresholds.

### 3.2.6 Threshold selection
We select the optimal threshold through cross-evaluation, namely using the threshold that yields the best performance on one TREC-DL dataset and applying it to another:
19→20, 20→19, 21→22, 22→21, and 22→23.
Run the following commands to perform the cross-evaluation:
```bash
JUDGES=(
  monot5-base
  monot5-large
  monot5-3b
  rankllama-7b
  rankllama-13b
  rank1-7b
  rank1-14b
  rank1-32b-awq
)

PAIRS=(
  "19,20"
  "21,22"
  "22,23"
)

SYS_DIR="./result/rq1_2/system_ranking"
RJ_DIR="./result/rq1_2/rj"
OUT_CSV="./result/rq1_2/cross_eval_summary.csv"

rm -f "$OUT_CSV"

suffix_for_dl() {
  local d=$1
  if [[ "$d" == "22" || "$d" == "23" ]]; then
    echo "passage_duped"
  else
    echo "passage"
  fi
}

for pair in "${PAIRS[@]}"; do
  IFS=',' read -r src tgt <<< "$pair"

  for judge in "${JUDGES[@]}"; do
    echo ">>> [DL${src}->DL${tgt}] judge=${judge}"

    src_suffix=$(suffix_for_dl "$src")
    tgt_suffix=$(suffix_for_dl "$tgt")

    src_sys="${SYS_DIR}/result.${judge}.dl${src}-${src_suffix}.json"
    tgt_sys="${SYS_DIR}/result.${judge}.dl${tgt}-${tgt_suffix}.json"
    src_rj="${RJ_DIR}/result.${judge}.dl${src}-${src_suffix}.json"
    tgt_rj="${RJ_DIR}/result.${judge}.dl${tgt}-${tgt_suffix}.json"

    for dir in "${src}->${tgt}" "${tgt}->${src}"; do
      if [[ "$dir" == "${src}->${tgt}" ]]; then
        s_sys="$src_sys"; t_sys="$tgt_sys"
        s_rj="$src_rj"; t_rj="$tgt_rj"
      else
        s_sys="$tgt_sys"; t_sys="$src_sys"
        s_rj="$tgt_rj"; t_rj="$src_rj"
      fi

      for m in ndcg@10 map@100 mrr@10; do
        python -u cross_evaluation.py \
          --src_json "$s_sys" \
          --tgt_json "$t_sys" \
          --task system_ranking \
          --metric "$m" \
          --judge "$judge" \
          --direction "DL${dir}" \
          --csv "$OUT_CSV"
      done

      python -u cross_evaluation.py \
        --src_json "$s_rj" \
        --tgt_json "$t_rj" \
        --task rj \
        --judge "$judge" \
        --direction "DL${dir}" \
        --csv "$OUT_CSV"
    done
  done
done

echo "Done. Results written to: $OUT_CSV"
```
The above command will generate the `cross_eval_summary.csv` file that records the selected threshold for each dataset under each target metric.


### 3.3 RQ3: Bias of re-ranker-based judges towards re-rankers

Before going into details, we need to create run folders for each dataset; each folder contains runs of BM25 and BM25 with re-rankers:
```bash
k=1000
for d in 19 20 21 22 23
do
mkdir ./output/rq3/runs/runs.bm25-${k}--reranker.dl${d}-passage # each dataset has a run folder
done
```

#### 3.3.1 Run BM25 for retrieval
Run the following commands run BM25 for retrieval (return 1000 documents) on TREC-DL 19 to 23：
```bash
k=1000
for d in 19 20 21 22 23
do
  if [[ $d -eq 19 ]]; then
      query_name="topics.dl${d}-passage.txt"
  else
      query_name="topics.dl${d}.txt"
  fi

  if [[ $d -eq 21 || $d -eq 22 || $d -eq 23 ]]; then
      index_name="lucene-index.msmarco-v2-passage-full.20220808.4d6d2a"
  else
      index_name="lucene-index.msmarco-v1-passage-full.20221004.252b5e"
  fi

  python -m pyserini.search.lucene \
    --threads 16 --batch-size 128 \
    --index ./data/indexes/${index_name} \
    --topics ./data/queries/${query_name} \
    --output ./output/rq3/runs/runs.bm25-${k}--reranker.dl${d}-passage/run.bm25-${k}.dl${d}-passage.txt \
    --bm25 --k1 0.9 --b 0.4 --hits ${k}
done
```
The above commands will produce BM25 run files on each dataset under `./output/rq3/runs/`.

#### 3.3.2 Generate re-ranker input files
Run the following commands to generate re-ranker input files for all re-rankers:
```bash
retriever=bm25-1000
for d in 19 20 21 22 23
do
    if [[ $d -eq 19 || $d -eq 20 ]]; then
        corpus_name="msmarco_v1_passage.tsv"
    else
        corpus_name="msmarco_v2_passage"
    fi

    if [[ $d -eq 19 ]]; then
        query_name="topics.dl${d}-passage.txt"
    else
        query_name="topics.dl${d}.txt"
    fi

    if [[ $d -eq 22 || $d -eq 23 ]]; then
        qrels_name="qrels.nist.dl${d}-passage_duped.txt"
    else
        qrels_name="qrels.nist.dl${d}-passage.txt"
    fi

    run_name="run.${retriever}.dl${d}-passage.txt"
    output_name="rerank_input.${retriever}.dl${d}-passage.jsonl"

    echo dl${d}, ${run_name}, ${qrels_name}, ${corpus_name}, ${query_name},${output_name}
    python prepare_rerank_file.py \
        --run_path ./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${run_name} \
        --qrels_path ./data/qrels/${qrels_name} \
        --corpus_path ./data/corpus/${corpus_name} \
        --query_path ./data/queries/${query_name} \
        --output_path ./output/rq3/rerank_input/${output_name}
done
```
The generated re-ranker input files will be saved in `./output/rq3/rerank_input/`.

#### 3.3.3 Run re-rankers on top of BM25

Run the following commands to re-rank the top 1,000 documents retrieved by BM25 using monoT5, Rank1, and RankLLaMA:
```bash
# BM25+monoT5
GPU_ID=0
BATCH_SIZE=8
retriever="bm25-1000" 

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for m in "castorini/monot5-base-msmarco" "castorini/monot5-large-msmarco" "castorini/monot5-3b-msmarco"
do
for d in 19 20 21 22 23
do
   echo ">>> Starting model ${m} on dataset dl${d} at $(date)"

   dataset_name="rerank_input.${retriever}.dl${d}-passage.jsonl"
   rerank_output_name="run.${retriever}--${m##*/}.dl${d}-passage.txt"
   qrels_output_name="qrels.${retriever}--${m##*/}-gen.dl${d}-passage.txt"
    
python -u monot5.py \
--model_name_or_path "${m}" \
--tokenizer_name "${m}" \
--dataset_path ./output/rq3/rerank_input/${dataset_name} \
--rerank_output_path ./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${rerank_output_name} \
--qrels_output_path ./output/rq3/qrels/${qrels_output_name} \
--batch_size 8
echo ">>> Finish model ${m} on dataset dl${d} at $(date)"
done
done

# BM25+Rank1
GPU_ID=0
BATCH_SIZE=512
retriever="bm25-1000" 

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for m in "jhu-clsp/rank1-7b" "jhu-clsp/rank1-14b" "jhu-clsp/rank1-32b-awq"
do
for d in 19 20 21 22 23
do
   echo ">>> Starting model ${m} on dataset dl${d} at $(date)"

   dataset_name="rerank_input.${retriever}.dl${d}-passage.jsonl"
   rerank_output_name="run.${retriever}--${m##*/}.dl${d}-passage.txt"
   qrels_output_name="qrels.${retriever}--${m##*/}-gen.dl${d}-passage.txt"

python -u rank1.py \
--model_name_or_path "${m}" \
--dataset_path ./output/rq3/rerank_input/${dataset_name} \
--rerank_output_path ./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${rerank_output_name} \
--qrels_output_path ./output/rq3/qrels/${qrels_output_name} \
--batch_size $BATCH_SIZE
echo ">>> Finish model ${m} on dataset dl${d} at $(date)"
done
done

# BM25+RankLLaMA
GPU_ID=0
BATCH_SIZE=32
retriever="bm25-1000" 

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"

tokenizers=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
)

models=(
  "castorini/rankllama-v1-7b-lora-passage"
  "castorini/rankllama-v1-13b-lora-passage"
)

for i in "${!models[@]}"; do
  t="${tokenizers[$i]}"
  m="${models[$i]}"

  for d in 19 20 21 22 23; do
    echo ">>> Starting model ${m} on dataset dl${d} at $(date)"

    dataset_name="rerank_input.${retriever}.dl${d}-passage.jsonl"
    rerank_output_name="run.${retriever}--${m##*/}.dl${d}-passage.txt"
    tmp_name="tmp_run.${retriever}--${m##*/}.dl${d}-passage.txt"

    python -m tevatron.reranker.driver.rerank \
      --output_dir=temp \
      --model_name_or_path "$m" \
      --tokenizer_name "$t" \
      --dataset_path "./output/rq3/rerank_input/${dataset_name}" \
      --fp16 \
      --per_device_eval_batch_size "${BATCH_SIZE}" \
      --rerank_max_len $((32 + 164)) \
      --dataset_name json \
      --query_prefix "query: " \
      --passage_prefix "document: " \
      --rerank_output_path "./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${tmp_name}"

    python -m tevatron.utils.format.convert_result_to_trec \
      --input "./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${tmp_name}" \
      --output "./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${rerank_output_name}"

    rm -f "./output/rq3/runs/runs.${retriever}--reranker.dl${d}-passage/${tmp_name}"

    echo ">>> Finished model ${m} on dataset dl${d} at $(date)"
  done
done
```
The output run files for TREC-DL 2019–2023 are saved under `./output/rq3/runs/` in per-dataset subfolders named `runs.bm25-1000--reranker.dl{yy}-passage/` (e.g., `…/runs.bm25-1000--reranker.dl19-passage/`, `…/runs.bm25-1000--reranker.dl20-passage/`).

#### 3.3.4 Measure the ranking performance of the BM25 runs re-ranked by monoT5, Rank1, and RankLLaMA
We use human, UMBRELLA and all re-ranker-based judges to measure the ranking performance of BM25–monoT5, BM25–Rank1, and BM25–RankLLaMA runs.

We first use human-annotated labels to measure ranking performance of all run files:
```bash
judge=nist

for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./output/rq3/runs/runs.bm25-1000--reranker.dl${d}-passage \
--qrels_dir ./data/qrels/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq3/ranking_performance \
--rel_scale 2
done
```
Files with ranking performance saved in `./output/rq3/ranking_performance`.

Also, we measure ranking performance of all run files using UMBRELLA for refereance:
```bash
judge=gpt-4o_0123_100_0_1

for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./output/rq3/runs/runs.bm25-1000--reranker.dl${d}-passage \
--qrels_dir ./output/rq1_2/qrels/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq3/ranking_performances \
--rel_scale 2
done
```

We then measure the ranking performance of all runs using monoT5 and Rank1 as evaluators (re-ranker-based judges).
Both evaluators use relevance judgments under the direct generation mode described in 3.1.
Please run the following commands:
```bash
for judge in monot5-base-gen monot5-large-gen monot5-3b-gen rank1-7b-gen rank1-14b-gen rank1-32b-awq-gen
do
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
         judgeset="dl${d}-passage"

    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_ranking_performance.py \
--run_dir ./output/rq3/runs/runs.bm25-1000--reranker.dl${d}-passage \ 
--qrels_dir ./output/rq1_2/qrels/qrels.${judge}.${judgeset}.txt \
--output_dir ./output/rq3/ranking_performances \
--rel_scale 1

done
done
```

For RankLLaMA, we use the relevance judgments produced with the thresholds (MAP@100 as the target metric) selected in 3.2 for each dataset; see `cross_eval_summary.csv` produeced in 3.2 for more infomration.
Run the following commands:
```bash
# rankllama-7b
cp -r ./output/rq1_2/qrels/qrels.rankllama-7b.dl19-passage/1.6.txt ./output/rq1_2/qrels/qrels.rankllama-7b-map100thr.dl19-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-7b.dl20-passage/1.2.txt ./output/rq1_2/qrels/qrels.rankllama-7b-map100thr.dl20-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-7b.dl21-passage/1.9.txt ./output/rq1_2/rels/qrels.rankllama-7b-map100thr.dl21-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-7b.dl22-passage_duped/3.3.txt ./output/rq1_2/qrels/qrels.rankllama-7b-map100thr.dl22-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-7b.dl23-passage_duped/1.9.txt ./output/rq1_2/qrels/qrels.rankllama-7b-map100thr.dl23-passage.txt

for d in 19 20 21 22 23
do
python -u evaluate_ranking_performance.py \
--run_dir ./output/runs/runs.bm25-1000--reranker.dl${d}-passage \
--qrels_dir ./output/rq1_2/qrels/qrels.rankllama-7b-map100thr.dl${d}-passage.txt \
--output_dir ./output/rq3 \
--rel_scale 1
done

# rankllama-13b
cp -r ./output/rq1_2/qrels/qrels.rankllama-13b.dl19-passage/3.4.txt ./output/rq1_2/qrels/qrels.rankllama-13b-map100thr.dl19-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-13b.dl20-passage/3.4.txt ./output/rq1_2/qrels/qrels.rankllama-13b-map100thr.dl20-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-13b.dl21-passage/3.7.txt ./output/rq1_2/rels/qrels.rankllama-13b-map100thr.dl21-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-13b.dl22-passage_duped/4.6.txt ./output/rq1_2/qrels/qrels.rankllama-13b-map100thr.dl22-passage.txt
cp -r ./output/rq1_2/qrels/qrels.rankllama-13b.dl23-passage_duped/3.7.txt ./output/rq1_2/qrels/qrels.rankllama-13b-map100thr.dl23-passage.txt

for d in 19 20 21 22 23
do
python -u evaluate_ranking_performance.py \
--run_dir ./output/runs/runs.bm25-1000--reranker.dl${d}-passage \
--qrels_dir ./output/rq1_2/qrels/qrels.rankllama-13b-map100thr.dl${d}-passage.txt \
--output_dir ./output/rq3 \
--rel_scale 1
done
```
Files with ranking performance saved in `./output/rq3/ranking_performance`.

#### 3.3.5 System ranking evaluation
We first evaluate the correlation between the system rankings produced by UMBRELLA and those produced by human judges for reference:
```bash
judge=gpt-4o_0123_100_0_1
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_system_ranking.py \
--a_dir ./output/rq3/ranking_performance/ranking_performance.nist.${judgeset}.json \
--b_dir ./output/rq3/ranking_performance_/ranking_performance.${judge}.${judgeset}.json \
--output_path ./result/rq3/system_ranking/result.${judge}.${judgeset}.json \
--metrics ndcg@10 map@100 mrr@10
done
```
The result files saved in `./result/rq3/system_ranking/`.

We then evaluate the correlation between the system rankings produced by the re-ranker-based judges and those produced by each re-ranker-based relevance judge:
```bash
for judge in monot5-base-gen monot5-large-gen monot5-3b-gen rankllama-7b-map100thr rankllama-13b-map100thr rank1-7b-gen rank1-14b-gen rank1-32b-awq-gen
do
for d in 19 20 21 22 23
do

 if [[ $d -eq 22 || $d -eq 23 ]]; then
        judgeset="dl${d}-passage_duped"
    else
        judgeset="dl${d}-passage"
    fi

echo ">>> ${judge} on ${judgeset}"
python -u evaluate_system_ranking.py \
--a_dir ./output/rq3/ranking_performance/ranking_performance.nist.${judgeset}.json \
--b_dir ./output/rq3/ranking_performance/ranking_performance.${judge}.${judgeset}.json \
--output_path ./result/rq3/system_ranking/result.${judge}.${judgeset}.json \
--metrics ndcg@10 map@100 mrr@10
done
done
```

## 4. Create plots
Run `rq2_thresholding.ipynb` to reproduce all plots for RQ2 (Re-rankers as judges via score thresholding) presented in the paper.
The reproduced plots will be saved in the `./plots/rq1_2` directory.

Run `rq3_bias.ipynb` to reproduce all plots for RQ3 (Bias of re-ranker-based judges towards re-rankers) presented in the paper.
The reproduced plots will be saved in the `./plots/rq3` directory.

### 5. Results on nDCG@10 (We report results on ndcg10 here because of limited space in the paper)
Due to limited space in our paper, we present results on nDCG@10 here.

nDCG@10 results for **RQ1**.
the following table shows Kendall’s τ correlation coefficients between the system orderings induced by relevance judgments from TREC assessors and those predicted by each re-ranker-based relevance judge adapted via **direct generation**.
Results are shown for system orderings based on nDCG@10.
The results of UMBRELA, a state-of-the-art relevance judge, are included for reference.
The best value in each column is **boldfaced**, while the second best is *italicized*.

| Method        | TREC-DL 19 | TREC-DL 20 | TREC-DL 21 | TREC-DL 22 | TREC-DL 23 |
| ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| UMBRELA       | 0.898      | **0.944**  | **0.936**  | *0.911*    | *0.929*    |
| monoT5 base   | 0.918      | 0.746      | 0.795      | 0.669      | 0.661      |
| monoT5 large  | *0.906*    | 0.792      | 0.841      | 0.708      | 0.737      |
| monoT5 3B     | **0.927**  | 0.845      | 0.902      | 0.783      | 0.723      |
| Rank1-7B      | 0.853      | 0.885      | 0.925      | 0.875      | **0.933**  |
| Rank1-14B     | 0.838      | 0.897      | *0.929*    | **0.912**  | 0.919      |
| Rank1-32B     | 0.862      | *0.909*    | 0.926      | 0.908      | 0.926      |

nDCG@10 results for **RQ2**.
The following table shows Kendall’s τ correlation coefficients between the system orderings induced by relevance judgments from TREC assessors and those predicted by each re-ranker-based relevance judge adapted via **score thresholding**.

| Method        | TREC-DL 19 | TREC-DL 20 | TREC-DL 21 | TREC-DL 22 | TREC-DL 23 |
| ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| UMBRELA       | 0.898      | **0.944**  | **0.936**  | *0.911*    | **0.929**  |
| monoT5 base   | *0.921*    | 0.738      | 0.786      | 0.713      | 0.435      |
| monoT5 large  | 0.900      | 0.789      | 0.814      | 0.715      | 0.734      |
| monoT5 3B     | **0.923**  | 0.816      | 0.851      | 0.802      | 0.771      |
| RankLLaMA-7B  | 0.819      | 0.861      | 0.848      | 0.802      | 0.649      |
| RankLLaMA-13B | 0.810      | 0.843      | 0.841      | 0.744      | 0.671      |
| Rank1-7B      | 0.853      | 0.867      | 0.925      | 0.878      | **0.929**  |
| Rank1-14B     | 0.859      | *0.912*    | *0.928*    | 0.912      | *0.926*    |
| Rank1-32B     | 0.850      | 0.910      | 0.922      | **0.919**  | **0.929**  |


Overall, our observations are similar to those obtained using MAP@100 or MRR@10 as target metrics.
Re-ranker-based judges (Rank1 and monoT5) show strong performance, , performing on par with or better than UMBRELA on three out of five datasets.