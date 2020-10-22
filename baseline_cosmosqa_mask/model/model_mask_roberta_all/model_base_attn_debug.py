import sys
import time
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

sys.path.append("../../../")
sys.path.append("../../../baseline_cosmosqa_mask/run_roberta_transformer_ensemble/modeling_roberta")
sys.path.append("../../../baseline_cosmosqa_mask/run_roberta_transformer_ensemble/model_transformer_fix_head")

from baseline_cosmosqa_mask.model.modeling_roberta import RobertaModel
from baseline_cosmosqa_mask.model.modeling_roberta import BertPreTrainedModel

from baseline_cosmosqa_mask.model.model_transformer import Constants
from baseline_cosmosqa_mask.model.model_transformer.Layers import EncoderLayer
from baseline_cosmosqa_mask.model.model_transformer.SubLayers import MultiHeadAttention
from baseline_cosmosqa_mask.model.model_transformer.SubLayers import PositionwiseFeedForward
# from baseline_cosmosqa_mask.model.model_transformer.Models import get_non_pad_mask
# from baseline_cosmosqa_mask.model.model_transformer.Models import get_attn_key_pad_mask


def get_non_pad_mask(seq):
    """
        sequence: [I, love, github, <pad>, <pad>]
        padding_mask: [1, 1, 1, 0, 0]
        padding_mask:
        [
            [1],
            [1],
            [1],
            [0],
            [0]
        ]
    """
    assert seq.dim() == 2
    return seq.ne(1).type(torch.float).unsqueeze(-1)


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


class Trans_Encoder_layer(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

        enc_slf_attn_list = []

        batch_size = enc_output.size(0)
        seq_len = enc_output.size(1)
        hidden_size = enc_output.size(2)

        for idx, enc_layer in enumerate(self.layer_stack):
            _slf_attn_mask = slf_attn_mask[idx]
            # _slf_attn_mask = _slf_attn_mask.byte()

            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=_slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = enc_output.view(-1, hidden_size)
        enc_output = self.bn(enc_output)
        enc_output = enc_output.view(batch_size, seq_len, hidden_size)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Trans_Encoder_self_attn(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.n_layer = int(n_layers)
        self.n_head = int(n_head)
        self.hidden_size = int(d_model)
        self.head_size = int(self.hidden_size / self.n_head)
        self.seq_len = 256

        self.linear = nn.Linear(self.hidden_size, self.n_head * 4)
        self.softmax = nn.Softmax(dim=-1)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

        batch_size = enc_output.size(0)
        seq_len = enc_output.size(1)

        enc_slf_attn_list = []

        for idx, enc_layer in enumerate(self.layer_stack):

            # 利用roberta_attn学习参数
            # roberta_attn: (batch_size * choice_num) * seq_len * hidden

            # mean pooling
            # weight: (batch_size * choice_num) * 1 * hidden_size
            weight = torch.sum(enc_output, dim=1) / self.seq_len  # 通过将Transformer的输出作为参数

            # weight: (batch_size * choice_num) * (n_head * 4)
            weight = self.linear(weight)

            # weight: (batch_size * choice_num) * n_head * 4
            weight = weight.view(batch_size, self.n_head, 4)

            # 不同的layer对应不同的mask矩阵
            # weight: (batch_size * choice_num) * n_head * 4
            weight = self.softmax(weight)

            # weight: (batch_size * choice_num) * n_head * 4 * 1 * 1
            weight = weight.unsqueeze(-1).unsqueeze(-1)

            # slf_attn_mask: (batch_size * choice_num) * 4 * seq_len * seq_len
            # slf_attn_mask: (batch_size * choice_num) * n_head * 4 * seq_len * seq_len

            _slf_attn_mask = slf_attn_mask.unsqueeze(1)
            _slf_attn_mask = _slf_attn_mask.repeat(1, 12, 1, 1, 1)

            _slf_attn_mask = torch.mul(weight.float(), _slf_attn_mask.float())

            # slf_attn_mask: (batch_size * choice_num) * n_head * seq_len * seq_len
            _slf_attn_mask = torch.sum(_slf_attn_mask, dim=2)

            # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
            _slf_attn_mask = _slf_attn_mask.view(-1, seq_len, seq_len)
            _slf_attn_mask = _slf_attn_mask.byte()

            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=_slf_attn_mask)
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


class Roberta_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(Roberta_Encoder, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):
        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]
        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = None

        flat_attention_mask   = attention_mask.view(  -1, seq_len)

        flat_position_ids    = None
        flat_head_mask       = None

        # roberta_attn:  [batch_size * choice_num, seq_len, hidden]
        # pooled_output: [batch_size * choice_num, hidden]
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        roberta_attn = roberta_attn.view(batch_size, num_choices, seq_len, -1)
        pooled_output = pooled_output.view(batch_size, num_choices, -1)
        return roberta_attn, pooled_output


class TransformerForMultipleChoice_Fusion_Layer(nn.Module):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + Sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fusion layer".format(time.ctime(time.time())))
        super(TransformerForMultipleChoice_Fusion_Layer, self).__init__()

        self.n_head = int(config.num_attention_heads)
        self.n_layer = 3
        self.hidden_size = int(config.hidden_size)
        self.layer_size  = int(self.hidden_size / self.n_layer)

        self.transformer_mrc = Trans_Encoder_layer(n_layers=3,
                                                   n_head=12,
                                                   d_k=64,
                                                   d_v=64,
                                                   d_model=768,
                                                   d_inner=4096,
                                                   dropout=0.1)

        self.pooler = BertPooler(config)

        self.linear  = nn.Linear(self.hidden_size, self.n_layer * 4)
        self.softmax = nn.Softmax(dim=-1)

        self.bn         = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self,
                input_ids,
                roberta_attn,
                pooled_output,

                commonsense_mask=None,
                dependency_mask=None,
                entity_mask=None,
                sentiment_mask=None,

                labels=None):
        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]
        hidden_size = roberta_attn.size(-1)

        entity_mask = entity_mask.unsqueeze(2).expand(batch_size, num_choices, seq_len, seq_len)

        flat_input_ids        = input_ids.view(-1, input_ids.size(-1))
        flat_commonsense_mask = commonsense_mask.view(-1, 1, seq_len, seq_len)
        flat_dependency_mask  = dependency_mask.view( -1, 1, seq_len, seq_len)
        flat_entity_mask      = entity_mask.view(     -1, 1, seq_len, seq_len)
        flat_sentiment_mask   = sentiment_mask.view(  -1, 1, seq_len, seq_len)

        # roberta_attn: (batch_size * choice_num) * seq_len * hidden_size
        roberta_attn = roberta_attn.view(-1, seq_len, hidden_size)

        # 利用Roberta_attn 的mean pooling学习参数
        # weight: (batch_size * choice_num) * hidden_size
        weight = torch.sum(roberta_attn, dim=1) / seq_len

        # weight: (batch_size * choice_num) * (n_layer * 4)
        weight = self.linear(weight)

        # weight: (batch_size * choice_num) * n_layer * 4
        weight = weight.view(batch_size * num_choices, self.n_layer, 4)

        # weight: n_layer * (batch_size * choice_num) * 4
        weight = weight.permute(1, 0, 2)

        # 不同的layer对应不同的mask矩阵
        # weight: n_layer * (batch_size * choice_num) * 4
        weight = self.softmax(weight)

        # weight: n_layer * (batch_size * choice_num) * 4 * 1 * 1
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        if   torch.get_default_dtype() == torch.float32: weight = weight.float()
        elif torch.get_default_dtype() == torch.float64: weight = weight.double()
        else: raise TypeError

        # flat_mask: (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = torch.cat([flat_commonsense_mask, flat_dependency_mask, flat_entity_mask, flat_sentiment_mask], dim=1)
        if   torch.get_default_dtype() == torch.float32: flat_mask = flat_mask.float()
        elif torch.get_default_dtype() == torch.float64: flat_mask = flat_mask.double()
        else: raise TypeError

        # flat_mask: n_layer * (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = flat_mask.unsqueeze(0).repeat(self.n_layer, 1, 1, 1, 1)
        flat_mask = torch.mul(weight, flat_mask)

        # slf_attn_mask: n_layer * (batch_size * choice_num) * seq_len * seq_len
        slf_attn_mask = torch.sum(flat_mask, dim=2)

        # slf_attn_mask: n_layer * (batch_size * choice_num) * n_head * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.unsqueeze(2).repeat(1, 1, self.n_head, 1, 1)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num) * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        if   torch.get_default_dtype() == torch.float32: non_pad_mask = non_pad_mask.float()
        elif torch.get_default_dtype() == torch.float64: non_pad_mask = non_pad_mask.double()
        else: raise TypeError

        # slf_attn_mask: n_layer * (batch_size * choice_num * n_head) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(self.n_layer, -1, seq_len, seq_len)

        tran_attn = self.transformer_mrc(enc_output=roberta_attn,
                                         non_pad_mask=non_pad_mask,
                                         slf_attn_mask=slf_attn_mask)

        sequence_output = roberta_attn + tran_attn
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
