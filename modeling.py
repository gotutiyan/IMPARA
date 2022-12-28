from transformers import AutoModel, AutoTokenizer, AutoConfig, BertForSequenceClassification, PreTrainedTokenizer
import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
from typing import List
from tqdm import tqdm
import math

class SimilarityEstimator(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_state = self.model(
            src_input_ids,
            src_attention_mask
        ).last_hidden_state
        pred_state = self.model(
            pred_input_ids,
            pred_attention_mask
        ).last_hidden_state
        src_pooler = self.mean_pooling(src_state, src_attention_mask)
        trg_pooler = self.mean_pooling(pred_state, pred_attention_mask)
        cosine_sim = CosineSimilarity()
        similarity = cosine_sim(src_pooler, trg_pooler)
        return similarity

    def mean_pooling(self, logits, mask):
        logits[mask == 0] = 0 # batch x seq_len x hidden
        sum_logits = torch.sum(logits, dim=1) # batch x hidden
        length = torch.sum(mask, dim=-1) # batch x
        pooled_logits = torch.div(sum_logits.transpose(1, 0), length).transpose(1, 0) # batch x hidden
        return pooled_logits

class QualityEstimatorForTrain(nn.Module):
    def __init__(self, qe_model: BertForSequenceClassification):
        super().__init__()
        self.model = qe_model
    
    def forward(
        self,
        low_input_ids: torch.Tensor,
        low_attention_mask: torch.Tensor,
        high_input_ids: torch.Tensor,
        high_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        low_scores = self.model(
            low_input_ids,
            low_attention_mask
        ).logits
        high_scores = self.model(
            high_input_ids,
            high_attention_mask
        ).logits
        loss = torch.sigmoid(low_scores - high_scores)
        loss = torch.mean(loss)
        return loss

    
class IMPARA(nn.Module):
    def __init__(
        self,
        se_model: SimilarityEstimator,
        qe_model: BertForSequenceClassification,
        tokenizer: PreTrainedTokenizer,
        threshold: float=0.9,
        max_len: int = 128
    ):
        super().__init__()
        self.se_model = se_model.eval()
        self.qe_model = qe_model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_len = max_len
    
    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            se = self.se_model(
                src_input_ids,
                src_attention_mask,
                pred_input_ids,
                pred_attention_mask,
            ).view(-1)
            qe = self.qe_model(
                pred_input_ids,
                pred_attention_mask
            ).logits.view(-1)
            qe = torch.sigmoid(qe)
            idx = se > self.threshold
            score = torch.zeros_like(se)
            score[idx] = qe[idx]
        return score
    
    def score(
        self,
        sources: List[str],
        predictions: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        scores = []
        for i in tqdm(range(math.ceil(len(sources) / batch_size))):
            src_encode = self.tokenizer(
                sources[i*batch_size:(i+1)*batch_size],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            pred_encode = self.tokenizer(
                predictions[i*batch_size:(i+1)*batch_size],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
            src_encode = {k:v.to(self.qe_model.device) for k,v in src_encode.items()}
            pred_encode = {k:v.to(self.qe_model.device) for k,v in pred_encode.items()}
            batch_score = self.forward(
                src_encode['input_ids'],
                src_encode['attention_mask'],
                pred_encode['input_ids'],
                pred_encode['attention_mask']
            ).view(-1)
            scores += batch_score.tolist()
        return scores