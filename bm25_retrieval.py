import argparse
import os
import logging
import json
import tqdm
from elasticsearch import Elasticsearch
import re
import sacrebleu
import editdistance
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--build_index', action='store_true',
        help='whether to train an index from scratch')
    parser.add_argument('--search_index', action='store_true',
        help='whether to search from a built index')
    parser.add_argument('--index_file', type=str, default='train.txt')
    parser.add_argument('--search_file', type=str, default='dev.txt')
    parser.add_argument('--output_file', type=str, default='out')
    parser.add_argument('--index_name', type=str)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--allow_hit', action='store_true')
    parser.add_argument('--resort', action='store_true')
    parser.add_argument('--task', type=str, default='domain_adaptation')


    return parser.parse_args()


def debpe(bpe):
    return re.sub(r'(@@ )|(@@ ?$)', '', bpe)

def get_unedited_words(src_sent, src_tm_sent):
    # Here first sentence is the edited sentence and the second sentence is the target sentence
    # If src_sent is the first sentence, then insertion and deletion should be reversed.
    """ Dynamic Programming Version of edit distance and finding unedited words."""
    a = src_sent.split()
    b = src_tm_sent.split()
    edit_distance = editdistance.eval(a, b)
    edit_distance = 1 - edit_distance / max(len(src_sent), len(src_tm_sent))

    return edit_distance

def get_topk_sent_id(src, src_sim, k=6):
    scores = list(map(lambda x: -get_unedited_words(src, x), src_sim))
    topk = sorted(zip(scores, range(len(scores))), key=lambda x: x[0])[:k]
    ans = [it[1] for it in topk]
    return ans

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.setLevel(logging.WARNING)

    es = Elasticsearch([{u'host': "localhost", u'port': "9200"}], timeout=30)
  
    if args.build_index:
        queries, responses = [], []
        with open(args.index_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if len(line.strip().split('\t')) < 2:
                    continue
                q, r = line.strip().split('\t')
                queries.append(q)
                responses.append(r)

        logger.info('build with elasticsearch')
        from elasticsearch.helpers import bulk
        body = {
            "settings": {
                "index": {
                    "analysis": {
                        "analyzer": "standard"
                    },
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                }
            },
            "mappings": {
                "properties": {
                    "query": {
                        "type": "text",
                        "similarity": "BM25",
                        "analyzer": "standard",
                    },
                    "response": {
                        "type": "text",
                    }
                }
            }
        }
        if es.indices.exists(index=args.index_name):
            es.indices.delete(index=args.index_name)
        es.indices.create(index=args.index_name, body=body)

        index = args.index_name

        if args.task == 'domain_adaptation':
            actions = []
            for idx, (query, response) in tqdm.tqdm(enumerate(zip(queries, responses)), total=len(queries)):
            
                action = {
                    "_index": index,
                    "_source": {
                        "query": query,
                        "response": response
                    }
                }
                actions.append(action)
                if len(actions) >= 10000:
                    success, _ = bulk(es, actions, raise_on_error=True)
                    actions = []
            if actions:
                success, _ = bulk(es, actions, raise_on_error=True)
            es.indices.refresh(index)
            info = es.indices.stats(index=index)
            print('total document indexed', info["indices"][index]["primaries"]["docs"]["count"]) 

    if args.search_index:
        queries, responses = [], []
        with open(args.search_file, 'r') as f:
            for line in f.readlines():
                q, r = line.strip().split('\t')
                queries.append(q)
                responses.append(r)

        logger.info('search with elasticsearch')

        index = args.index_name
        query_body = {
                "query": {
                "match":{
                     "query": None
                 }
                },
                "size": args.topk
        }

        with open(args.output_file,'w') as fo:
            for query, response in tqdm.tqdm(zip(queries, responses), total=len(queries)):
                query_body["query"]["match"]["query"] = query
                es_result = es.search(index=index, body=query_body)
                ret_q = [ item["_source"]["query"] for item in es_result["hits"]["hits"]]
                ret_r = [ item["_source"]["response"] for item in es_result["hits"]["hits"]]

                if not args.allow_hit:
                    ret_q = ret_q[1:]
                    ret_r = ret_r[1:]
 
                ids = get_topk_sent_id(query, ret_q, args.topk)

                if args.resort:
                    ret = [[ret_q[ids[i]], ret_r[ids[i]]] if i < len(ids) else ['1', '1'] for i in range(args.topk)]
                else:
                    ret = [[ret_q[i], ret_r[i]] if i < len(ids) else ['1', '1'] for i in range(args.topk)]

                ret = sum(ret, [])
                line = '\t'.join([query, response] + ret)
                fo.write(line+'\n')

                if args.task == 'online_learning':
                    action = {
                        "_index": index,
                        "_source": {
                            "query": query,
                            "response": response
                        }
                    }
                    success, _ = bulk(es, [action], raise_on_error=True)
                    es.indices.refresh(index)


if __name__ == "__main__":
    args = parse_args()
    main(args)
