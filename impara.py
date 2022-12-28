import argparse
from modeling import SimilarityEstimator, IMPARA
from transformers import AutoTokenizer, BertForSequenceClassification, AutoConfig
from dataset import Dataset
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import numpy as np
import random

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    qe_model = BertForSequenceClassification.from_pretrained(args.restore_dir)
    se_model = SimilarityEstimator('bert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    scorer = IMPARA(se_model, qe_model, tokenizer, threshold=args.threshold).cuda()
    srcs = open(args.src_file).read().rstrip().split('\n')
    preds = open(args.pred_file).read().rstrip().split('\n')
    scores = scorer.score(srcs, preds)
    if args.level == 'corpus':
        print(sum(scores) / len(scores))
    else:
        print(scores)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--restore_dir', default='gotutiyan/IMPARA-QE')
    parser.add_argument('--level', choices=['corpus', 'sentence'], default='corpus')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
