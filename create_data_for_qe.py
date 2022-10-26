
from transformers import AutoModel, AutoTokenizer
from torch.nn import CosineSimilarity
import argparse
import torch
import errant
import random
from typing import List
import pprint
import numpy as np
from tqdm import tqdm
from errant.edit import Edit

class Dataset:
    def __init__(self, sents: List[str]):
        self.sents = sents
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx: int):
        sent = self.sents[idx]
        
        encode = self.tokenizer.batch_encode_plus([sent], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        return {
            'input_ids': encode['input_ids'].squeeze().to(dtype=torch.long),
            'attention_mask': encode['attention_mask'].squeeze().to(dtype=torch.long),
        }

def generate_correction_from_edits(src: str, edits: List[Edit]) -> str:
    diff = 0
    tokens = src.split()
    for e in edits:
        if e.o_start == -1:
            continue
        for _ in range(e.o_start, e.o_end):
            try:
                del(tokens[e.o_start + diff])
            except IndexError:
                # print(f'{e.type=} {e.c_str=} {e.o_start+diff=} {e.o_end+diff=}\n{tokens=}')
                pass
        if e.type.startswith('U:'):
            diff -= len(e.o_str.split())
            continue

        corrected_tokens = reversed(e.c_str.split())
        for c_token in corrected_tokens:
            tokens.insert(e.o_start + diff, c_token)

        if e.type.startswith('M'):
            diff += len(e.c_str.split())
        elif e.type.startswith('R'):
            diff += len(e.c_str.split()) - len(e.o_str.split())
    return ' '.join(tokens)
    

def generate_except_each_edits(src: str, edits: List[Edit]) -> List[str]:
    trg_minus_sents = []
    for i, edit in enumerate(edits):
        applied_edits = edits[:]
        del(applied_edits[i])
        trg_minus = generate_correction_from_edits(src, applied_edits)
        trg_minus_sents.append(trg_minus)
    return trg_minus_sents
        

def mean_pooling(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits[mask == 0] = 0 # batch x seq_len x hidden
    sum_logits = torch.sum(logits, dim=1) # batch x hidden
    length = torch.sum(mask, dim=-1) # batch x
    pooled_logits = torch.div(sum_logits.transpose(1, 0), length).transpose(1, 0) # batch x hidden
    return pooled_logits

def edit_impact(
    trgs: List[str],
    trg_minus: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer
) -> torch.Tensor:
    trg_encode = tokenizer(trgs, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    trg_minus_encode = tokenizer(trg_minus, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    trg_encode = {k:v.cuda() for k, v in trg_encode.items()}
    trg_minus_encode = {k:v.cuda() for k, v in trg_minus_encode.items()}
    with torch.no_grad():
        trg_state = model(**trg_encode).last_hidden_state
        trg_minus_state = model(**trg_minus_encode).last_hidden_state
    cosine_sim = CosineSimilarity()
    return 1 - cosine_sim(
        mean_pooling(trg_state, trg_encode['attention_mask']),
        mean_pooling(trg_minus_state, trg_minus_encode['attention_mask'])
    )

def impact_of_subset(impacts: List[int], subset: List[int]) -> int:
    sum_impact = 0
    for i in subset:
        sum_impact += impacts[i]
    return sum_impact


def get_subset(n: int) -> List[int]:
    n_samples = random.randint(1, n)
    indices = [i for i in range(n)]
    random.shuffle(indices)
    return sorted(indices[:n_samples])

def get_another_subset(subset: List[int], n:int) -> List[int]:
    subset2 = subset[:]
    for i in range(n):
        do_sample = False
        if random.randint(1, n) == 1:
            do_sample = True
        if do_sample:
            if i in subset and i in subset2:
                del subset2[subset2.index(i)]
            elif i not in subset and i not in subset2:
                subset2.append(i)
    return sorted(subset2)
    

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    srcs = open(args.src).read().rstrip().split('\n')
    trgs = open(args.trg).read().rstrip().split('\n')
    annotator = errant.load('en')
    model = AutoModel.from_pretrained(args.model_id)
    model.cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    generated_pairs = []
    for src, trg in tqdm(zip(srcs, trgs), total=len(srcs)):
        if src == trg:
            continue
        sent_pairs = [] # [[low, high], [low, high]...[low, high]]
        parsed_src = annotator.parse(src)
        parsed_trg = annotator.parse(trg)
        edits = annotator.annotate(parsed_src, parsed_trg)
        trg_minus_sents = generate_except_each_edits(src, edits)
        
        impacts = edit_impact([trg] * len(trg_minus_sents), trg_minus_sents, model, tokenizer)
        impacts = impacts.tolist()
        for _ in range(args.n_try):
            subset1 = get_subset(len(edits))
            subset2 = get_another_subset(subset1, len(edits))
            if subset1 == subset2:
                continue
            impact1 = impact_of_subset(impacts, subset1)
            impact2 = impact_of_subset(impacts, subset2)
            edits1 = [edits[i] for i in subset1]
            edits2 = [edits[i] for i in subset2]
            sent1 = generate_correction_from_edits(src, edits1)
            sent2 = generate_correction_from_edits(src, edits2)
            # Put low-impact one on the left
            if impact1 > impact2:
                pair = [sent2, sent1]
            elif impact1 < impact2:
                pair = [sent1, sent2]
            if pair not in sent_pairs and pair[0] != '' and pair[1] != '':
                sent_pairs.append(pair)
        generated_pairs += sent_pairs
    random.shuffle(generated_pairs)
    for pairs in generated_pairs[:args.n_generated]:
        print('\t'.join(pairs))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--trg', required=True)
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--n_generated', default=4096, help='The number of training instances')
    parser.add_argument('--n_try', default=30, help='The number of times to try generate supervisoin data from a parallel data.')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)