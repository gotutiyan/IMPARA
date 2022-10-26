from transformers import AutoTokenizer
import torch
from typing import List, Tuple
from transformers import AutoTokenizer

class DatasetForTrainQE:
    def __init__(
        self,
        low_examples: List[str],
        high_examples: List[str],
        tokenizer: AutoTokenizer
    ):
        self.low_examples = low_examples
        self.high_examples = high_examples
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.low_examples)
    
    def __getitem__(self, idx: int):
        low = self.low_examples[idx]
        high = self.high_examples[idx]
        
        low_encode = self.tokenizer.batch_encode_plus([low], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        high_encode = self.tokenizer.batch_encode_plus([high], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        
        return {
            'low_input_ids': low_encode['input_ids'].squeeze().to(dtype=torch.long),
            'low_attention_mask': low_encode['attention_mask'].squeeze().to(dtype=torch.long),
            'high_input_ids': high_encode['input_ids'].squeeze().to(dtype=torch.long),
            'high_attention_mask': high_encode['attention_mask'].squeeze().to(dtype=torch.long)
        }

def generate_dataset(file_path: str, tokenizer: AutoTokenizer):
    low_impact_sents = []
    high_impact_sents = []
    with open(file_path) as fp:
        for line in fp:
            try:
                low, high = line.rstrip().split('\t')
            except:
                continue
            low_impact_sents.append(low)
            high_impact_sents.append(high)
    dataset = DatasetForTrainQE(
        low_impact_sents,
        high_impact_sents,
        tokenizer
    )
    return dataset

class Dataset:
    def __init__(self, srcs: List[str], preds: List[str], tokenizer: AutoTokenizer):
        self.srcs = srcs
        self.preds = preds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, idx: int):
        src = self.srcs[idx]
        pred = self.preds[idx]
        
        s_encode = self.tokenizer.batch_encode_plus([src], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        p_encode = self.tokenizer.batch_encode_plus([pred], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        
        return {
            'src_input_ids': s_encode['input_ids'].squeeze().to(dtype=torch.long),
            'src_attention_mask': s_encode['attention_mask'].squeeze().to(dtype=torch.long),
            'pred_input_ids': p_encode['input_ids'].squeeze().to(dtype=torch.long),
            'pred_attention_mask': p_encode['attention_mask'].squeeze().to(dtype=torch.long)
        }