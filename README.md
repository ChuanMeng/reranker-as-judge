# Reproducing Re-Rankers as Relevance Judges

This is the repository for the paper titled **Reproducing Re-Rankers as Relevance Judges**.
In this paper, we reproduce three re-rankers from different re-ranker families (monoT5, RankLLaMA, and Rank1) as relevance judgment predictors (a.k.a. relevance judges).

This repository is structured into five distinct parts:
- [1. Prerequest](#1-prerequest)
- [2. Data preparation](#2-data-preparation)
- [3. Reproducing results](#3-reproducing-results)
  - [3.1 RQ1: Re-rankers as judges via direct generation](#31-rq1-re-rankers-as-judges-via-direct-generation)
  - [3.2 RQ2: Re-rankers as judges via score thresholding](#32-rq2-re-rankers-as-judges-via-score-thresholding)
  - [3.3 RQ3: Bias of re-ranker-based judges towards re-rankers](#33-rq3-bias-of-re-ranker-based-judges-towards-re-rankers)
  - [3.4 Create plots](#34-create-plots)
  - [3.5 Results on nDCG@10](#35-results-on-ndcg10-we-report-the-results-on-ndcg10-here-because-of-limited-space-in-the-paper)

## 1. Prerequest
We recommend executing all processes in a Linux environment.
Install dependencies:
```bash
pip install -r requirements.txt
```
Also, please install [Tevatron](https://github.com/texttron/tevatron) and [Pyserini](https://github.com/castorini/pyserini) in advance.

## 2. Data preparation
### 2.1 Download folders
Create the following folders:
```bash
mkdir data
mkdir data/queries
mkdir data/qrels
mkdir data/indexes
mkdir data/corpus

mkdir output
mkdir output/rerank_input
mkdir output/qrels
mkdir output/runs
mkdir result
mkdir plots
```

### 2.2 Download quries, qrels and collections for TREC-DL
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

### 2.3 Download submitted run files for TREC-DL 19 to 21

### 2.3 Ask submitted run files for TREC-DL 22 and 23
Note that we follow 
Please ask Shivani Upadhyay, the first author of the paper [UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor](https://arxiv.org/pdf/2406.06519) to get the

```
## 3. Reproducing results

### 3.1 RQ1: Re-rankers as judges via direct generation

### 3.2 RQ2: Re-rankers as judges via score thresholding

### 3.3 RQ3: Bias of re-ranker-based judges towards re-rankers

### 3.4 Create plots
Run `rq2_thresholding.ipynb` to reproduce all plots for RQ2 (Re-rankers as judges via score thresholding) presented in the paper.
The reproduced plots will be saved in the `./plots/rq2` directory.

Run `rq3_bias.ipynb` to reproduce all plots for RQ3 (Bias of re-ranker-based judges towards re-rankers) presented in the paper.
The reproduced plots will be saved in the `./plots/rq3` directory.

### 3.5 Results on nDCG@10 (We report the results on nDCG@10 here because of limited space in the paper)
Due to limited space in our paper, we present results on nDCG@10 here.