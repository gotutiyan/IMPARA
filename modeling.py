from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from torch.nn import CosineSimilarity

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
    
class QualityEstimator(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        self.linear = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.bert(
            input_ids,
            attention_mask
        )
        cls_token = out.last_hidden_state[:, 0, :]
        pred = self.linear(cls_token)
        return pred

class QualityEstimatorForTrain(nn.Module):
    def __init__(self, qe_model: QualityEstimator):
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
        )
        high_scores = self.model(
            high_input_ids,
            high_attention_mask
        )
        loss = torch.sigmoid(low_scores - high_scores)
        loss = torch.mean(loss)
        return loss

    
class IMPARA(nn.Module):
    def __init__(
        self,
        se_model: SimilarityEstimator,
        qe_model: QualityEstimator,
        threshold: float=0.9
    ):
        super().__init__()
        self.se_model = se_model
        self.qe_model = qe_model
        self.threshold = threshold
    
    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        se = self.se_model(
            src_input_ids,
            src_attention_mask,
            pred_input_ids,
            pred_attention_mask,
        ).view(-1)
        qe = self.qe_model(
            pred_input_ids,
            pred_attention_mask
        ).view(-1)
        qe = torch.sigmoid(qe)
        idx = se > self.threshold
        score = torch.zeros_like(se)
        score[idx] = qe[idx]
        return score