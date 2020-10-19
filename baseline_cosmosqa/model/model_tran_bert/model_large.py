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
    ''' A encoder model with self attention mechanism. '''

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


# 用entity_mask的并集作为mask
class BertForMultipleChoice_Entity_Mask(BertPreTrainedModel):
    def __init__(self, config):
        print("entity_mask + dependency_mask")
        super(BertForMultipleChoice_Entity_Mask, self).__init__(config)

        self.bert = BertModel(config)
        self.transformer_mrc_e = Trans_Encoder(n_layers=3,
                                               n_head=4,
                                               d_k=192,
                                               d_v=192,
                                               d_model=768,
                                               d_inner=3072,
                                               dropout=0.1)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                entity_mask=None,
                labels=None):
        num_choices = input_ids.shape[1]
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
        flat_entity_mask     = entity_mask.view(-1, entity_mask.size(-1))
        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.bert(input_ids=flat_input_ids,
                                                token_type_ids=flat_token_type_ids,
                                                attention_mask=flat_attention_mask,
                                                position_ids=flat_position_ids,
                                                head_mask=flat_head_mask)

        # -- Prepare masks
        flat_entity_mask = flat_entity_mask
        non_pad_mask     = get_non_pad_mask(flat_input_ids)

        batch_size = flat_entity_mask.size(0)
        seq_len    = flat_entity_mask.size(1)

        # NOTE: input1, input2, output
        # output = input1*input2      # 数乘
        # output = input.mul(input2)  # 数乘
        # output = input.mm(input2)   # 矩阵乘

        # entity_mask_sum: (batch_size * choice_num) * seq_len
        # entity_mask = torch.Tensor(batch_size, seq_len).zero_().float().cuda()
        entity_mask_pad = flat_entity_mask.sum(1).eq(0).float().unsqueeze(1).expand(batch_size, seq_len)
        entity_mask_pad = entity_mask_pad.mul(flat_input_ids.ne(Constants.PAD).float())
        entity_mask_pad = entity_mask_pad.float().cuda()
        entity_mask_sum = flat_entity_mask.float() + entity_mask_pad.float()

        slf_attn_mask_keypad = entity_mask_sum.eq(Constants.PAD).unsqueeze(1).expand(batch_size, seq_len, -1).long()

        slf_attn_mask = slf_attn_mask_keypad
        tran_attn_e = self.transformer_mrc_e(enc_output=roberta_attn,
                                             non_pad_mask=non_pad_mask.float(),
                                             slf_attn_mask=slf_attn_mask.byte())

        # sequence_output = roberta_attn
        sequence_output = roberta_attn + tran_attn_e
        pooled_output = sequence_output[:, 0]
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
