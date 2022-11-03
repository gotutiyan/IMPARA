
import argparse
from modeling import SimilarityEstimator, IMPARA
from transformers import AutoTokenizer, BertForSequenceClassification
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

    config = json.load(open(os.path.join(args.restore_dir, 'impara_config.json')))
    model_id = config['model_id']
    qe_model = BertForSequenceClassification.from_pretrained(args.restore_dir)
    se_model = SimilarityEstimator(model_id)
    model = IMPARA(se_model, qe_model, threshold=args.threshold).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    srcs = open(args.src_file).read().rstrip().split('\n')
    preds = open(args.pred_file).read().rstrip().split('\n')
    dataset = Dataset(srcs, preds, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    scores = []
    model.eval()
    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            batch_score = model(**batch).view(-1).tolist()
            scores += batch_score
    if args.level == 'corpus':
        print(sum(scores) / len(scores))
    else:
        print(scores)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--pred_file', required=True)
    parser.add_argument('--restore_dir', required=True)
    parser.add_argument('--level', choices=['corpus', 'sentence'], default='corpus')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
