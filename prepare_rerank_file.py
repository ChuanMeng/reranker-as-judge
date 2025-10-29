import json
from argparse import ArgumentParser
from tqdm import tqdm
import pytrec_eval
import logging

def get_passage(pid,path):
    (string1, string2, bundlenum, position) = pid.split("_")
    assert string1 == "msmarco" and string2 == "passage"

    with open(f"{path}/msmarco_passage_{bundlenum}", "rt", encoding="utf8") as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document["pid"] == pid

        return document["passage"].replace("\t", "").replace("\n", "").replace("\r", "")

def load_corpus(path):
    corpus = dict()
    count=0
    with open(path, "r") as r:
        for line in tqdm(r):
            docid, text = line.strip().split("\t")
            corpus[docid]=text.replace("\t", "").replace("\n", "").replace("\r", "")
            count+=1
    return corpus


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True)
    parser.add_argument('--corpus_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument("--qrels_path", type=str, default=None)
    parser.add_argument("--run_path", type=str, default=None)

    args = parser.parse_args()


    qid2qtext={}
    with open(args.query_path, 'r') as r:
        for line in r.readlines():
            qid, qtext = line.split('\t')
            qid2qtext[qid]=qtext.replace("\t", "").replace("\n", "").replace("\r", "")

   #searcher = LuceneSearcher(args.corpus_path)
    if "v1" in args.corpus_path:
        corpus = load_corpus(args.corpus_path)


    if args.qrels_path is not None and args.run_path is None:
        with open(args.qrels_path, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        count_exp =0
        with open(args.output_path, 'w') as w:
            for qid, docid2rel in qrels.items():
                for docid, rel in docid2rel.items():
                    psg_info={}

                    #doc_dict = json.loads(searcher.doc(docid).raw())
                    #text = doc_dict['contents'] if 'contents' in doc_dict else doc_dict['passage']
                    if "v1" in args.corpus_path:
                        text = corpus[docid]
                    elif "v2" in args.corpus_path:
                        text = get_passage(docid, args.corpus_path)
                    else:
                        raise Exception

                    psg_info["docid"]= docid
                    psg_info["title"]= ""
                    psg_info["text"]= text

                    psg_info['score'] = 0
                    psg_info['query_id'] = qid
                    psg_info['query'] = qid2qtext[qid]

                    w.write(json.dumps(psg_info) + '\n')
                    count_exp += 1

        print(f"write in # {count_exp} examples")

    elif args.run_path is not None and args.qrels_path is not None:

        with open(args.qrels_path, 'r') as r:
            qrels = pytrec_eval.parse_qrel(r)

        with open(args.run_path, 'r') as r:
            run = pytrec_eval.parse_run(r)

        count_exp =0
        with open(args.output_path, 'w') as w:
            for qid, docid2score in run.items():

                if qid not in qrels.keys():
                    # do not give non-labelled queries to rerankers
                    continue

                for docid, score in docid2score.items():
                    psg_info={}
                    
                    if "v1" in args.corpus_path:
                        text = corpus[docid]
                    elif "v2" in args.corpus_path:
                        text = get_passage(docid, args.corpus_path)
                    else:
                        raise Exception

                    psg_info["docid"]= docid
                    psg_info["title"]= ""
                    psg_info["text"]= text

                    psg_info['score'] = score
                    psg_info['query_id'] = qid
                    psg_info['query'] = qid2qtext[qid]

                    w.write(json.dumps(psg_info) + '\n')
                    count_exp += 1

        print(f"write in # {count_exp} examples")

    else:
        raise Exception