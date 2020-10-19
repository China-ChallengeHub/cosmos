import os
import sys
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
gra_dir  = os.path.dirname(par_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
sys.path.append(gra_dir)

from pytorch_pretrained_xbert.modeling_bert import BertModel
from pytorch_pretrained_xbert.modeling_bert import BertPreTrainedModel

from baseline_cosmosqa.model.model_transformer import Constants
from baseline_cosmosqa.model.model_transformer.Layers import EncoderLayer
from baseline_cosmosqa.model.model_transformer.Models import get_non_pad_mask
from baseline_cosmosqa.model.model_transformer.Models import get_attn_key_pad_mask


class Trans_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        # enc_output = self.bn(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.transformer_mrc = Trans_Encoder(n_layers=3,
                                             n_head=12,
                                             d_k=64,
                                             d_v=64,
                                             d_model=768,
                                             d_inner=3072,
                                             dropout=0.1)

        self.pooler = BertPooler(config)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        # input_ids:      batch_size * choice_num * seq_len
        # token_type_ids: batch_size * choice_num * seq_len
        # attention_mask: batch_size * choice_num * seq_len
        # entity_mask:    batch_size * choice_num * seq_len

        # flat_input_ids:      (batch_size * choice_num) * seq_len
        # flat_token_type_ids: (batch_size * choice_num) * seq_len
        # flat_attention_mask: (batch_size * choice_num) * seq_len
        # flat_entity_mask:    (batch_size * choice_num) * seq_len

        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask  = attention_mask.view(-1, attention_mask.size(-1))
        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        bert_attn, pooled_output = self.bert(input_ids=flat_input_ids,
                                             token_type_ids=flat_token_type_ids,
                                             attention_mask=flat_attention_mask,
                                             position_ids=flat_position_ids,
                                             head_mask=flat_head_mask)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num) * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
        non_pad_mask  = get_non_pad_mask(flat_input_ids)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=flat_input_ids, seq_q=flat_input_ids)
        tran_attn = self.transformer_mrc(enc_output=bert_attn,
                                         non_pad_mask=non_pad_mask.float(),
                                         slf_attn_mask=slf_attn_mask.byte())

        # Debug: pooled_output分类器并非取自sequence_output[:, 0]
        # sequence_output = bert_attn + tran_attn
        # pooled_output = sequence_output[:, 0]

        sequence_output = bert_attn + tran_attn
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.bn(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # logits:          (batch_size * num_choices)
        # reshaped_logits: (batch_size,  num_choices)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss, reshaped_logits
        else:
            return reshaped_logits
