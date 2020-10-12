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

        self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

        enc_slf_attn_list = []

        batch_size = enc_output.size(0)
        seq_len = enc_output.size(1)
        hidden_size = enc_output.size(2)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = enc_output.view(-1, hidden_size)
        enc_output = self.bn(enc_output)
        enc_output = enc_output.view(batch_size, seq_len, hidden_size)

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

        # self.bn = torch.nn.BatchNorm1d(d_model)

    def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

        enc_slf_attn_list = []

        for idx, enc_layer in enumerate(self.layer_stack):
            _slf_attn_mask = slf_attn_mask[idx]
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=_slf_attn_mask.byte())
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


class RobertaForMultipleChoice_Fix_Head(BertPreTrainedModel):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fix head".format(time.ctime(time.time())))
        super(RobertaForMultipleChoice_Fix_Head, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=16, d_k=64, d_v=64,
                                             d_model=1024, d_inner=4096, dropout=0.1)

        self.pooler = BertPooler(config)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,

                attention_mask=None,
                commonsense_mask=None,
                dependency_mask=None,
                entity_mask=None,
                sentiment_mask=None,

                labels=None):

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        entity_mask = entity_mask.unsqueeze(2).expand(batch_size, num_choices, seq_len, seq_len)

        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = None

        flat_attention_mask   = attention_mask.view(  -1, seq_len)
        flat_commonsense_mask = commonsense_mask.view(-1, seq_len, seq_len)
        flat_dependency_mask  = dependency_mask.view( -1, seq_len, seq_len)
        flat_entity_mask      = entity_mask.view(     -1, seq_len, seq_len)
        flat_sentiment_mask   = sentiment_mask.view(        -1, seq_len, seq_len)

        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        # flat_attention_mask: (batch_size * choice_nums) * seq_len * seq_len
        flat_attention_mask = flat_attention_mask.unsqueeze(1).expand(-1, seq_len, seq_len)

        flat_attention_mask   = flat_attention_mask.view(  -1, 1, seq_len, seq_len)
        flat_commonsense_mask = flat_commonsense_mask.view(-1, 1, seq_len, seq_len)
        flat_dependency_mask  = flat_dependency_mask.view( -1, 1, seq_len, seq_len)
        flat_entity_mask      = flat_entity_mask.view(     -1, 1, seq_len, seq_len)
        flat_sentiment_mask   = flat_sentiment_mask.view(  -1, 1, seq_len, seq_len)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_nums) *      seq_len *    1
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        non_pad_mask = non_pad_mask.float()

        # slf_attn_mask: (batch_size * choice_nums) * 12 * seq_len * seq_len
        slf_attn_mask = torch.cat(
            [flat_commonsense_mask] * 4 + [flat_dependency_mask] * 4 +
            [flat_entity_mask] * 4 + [flat_sentiment_mask] * 4, dim=1)

        # slf_attn_mask: (batch_size * choice_nums * 12) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(-1, seq_len, seq_len)

        # fix_head时, 可加可不加
        slf_attn_mask = slf_attn_mask.byte()

        tran_attn = self.transformer_mrc(enc_output=roberta_attn,
                                         non_pad_mask=non_pad_mask,
                                         slf_attn_mask=slf_attn_mask)

        # Debug: pooled_output分类器并非取自sequence_output[:, 0]
        # sequence_output = bert_attn + tran_attn
        # pooled_output = sequence_output[:, 0]

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


class RobertaForMultipleChoice_Fusion_All(BertPreTrainedModel):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fusion all".format(time.ctime(time.time())))
        super(RobertaForMultipleChoice_Fusion_All, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder_layer(n_layers=3, n_head=16, d_k=64, d_v=64,
                                                   d_model=1024, d_inner=4096, dropout=0.1)

        self.pooler = BertPooler(config)

        self.weight = nn.Parameter(torch.randn(3, 12, 4))
        self.softmax = nn.Softmax(dim=-1)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,

                attention_mask=None,
                commonsense_mask=None,
                dependency_mask=None,
                entity_mask=None,
                sentiment_mask=None,

                labels=None):

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        entity_mask = entity_mask.unsqueeze(2).expand(batch_size, num_choices, seq_len, seq_len)

        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = None

        flat_attention_mask   = attention_mask.view(  -1, seq_len)
        flat_commonsense_mask = commonsense_mask.view(-1, seq_len, seq_len)
        flat_dependency_mask  = dependency_mask.view( -1, seq_len, seq_len)
        flat_entity_mask      = entity_mask.view(     -1, seq_len, seq_len)
        flat_sentiment_mask   = sentiment_mask.view(  -1, seq_len, seq_len)

        flat_position_ids    = None
        flat_head_mask       = None

        # roberta_attn:  (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        flat_commonsense_mask = flat_commonsense_mask.view(-1, 1, seq_len, seq_len)
        flat_dependency_mask  = flat_dependency_mask.view( -1, 1, seq_len, seq_len)
        flat_entity_mask      = flat_entity_mask.view(     -1, 1, seq_len, seq_len)
        flat_sentiment_mask   = flat_sentiment_mask.view(  -1, 1, seq_len, seq_len)

        # 不同的layer, 不同的head对应不同的mask矩阵
        # weight: 3 * 12 * 4
        weight = self.softmax(self.weight)

        # weight: 3 * 1 * 12 * 4 * 1 * 1
        weight = weight.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # flat_mask: (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = torch.cat(
            [flat_commonsense_mask, flat_dependency_mask, flat_entity_mask, flat_sentiment_mask], dim=1)

        # flat_mask: 3 * (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = flat_mask.unsqueeze(0).repeat(3, 1, 1, 1, 1)

        # flat_mask: 3 * (batch_size * choice_num) * 12 * 4 * seq_len * seq_len
        flat_mask = flat_mask.unsqueeze(2).repeat(1, 1, 12, 1, 1, 1)
        flat_mask = torch.mul(weight.float(), flat_mask.float())

        # slf_attn_mask: 3 * (batch_size * choice_num) * 12 * seq_len * seq_len
        slf_attn_mask = torch.sum(flat_mask, dim=3)

        # slf_attn_mask: 3 * (batch_size * choice_num * 12) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(3, -1, seq_len, seq_len)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num)      * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * 12 * seq_len * seq_len
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        non_pad_mask = non_pad_mask

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


class RobertaForMultipleChoice_Fusion_Head(BertPreTrainedModel):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fusion head".format(time.ctime(time.time())))
        super(RobertaForMultipleChoice_Fusion_Head, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=16, d_k=64, d_v=64,
                                             d_model=1024, d_inner=4096, dropout=0.1)

        self.pooler = BertPooler(config)

        self.weight = nn.Parameter(torch.FloatTensor(12, 4))
        self.softmax = nn.Softmax(dim=-1)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,

                attention_mask=None,
                commonsense_mask=None,
                dependency_mask=None,
                entity_mask=None,
                sentiment_mask=None,

                labels=None):

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        entity_mask = entity_mask.unsqueeze(2).expand(batch_size, num_choices, seq_len, seq_len)

        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = None

        flat_attention_mask   = attention_mask.view(  -1, seq_len)
        flat_commonsense_mask = commonsense_mask.view(-1, seq_len, seq_len)
        flat_dependency_mask  = dependency_mask.view( -1, seq_len, seq_len)
        flat_entity_mask      = entity_mask.view(     -1, seq_len, seq_len)
        flat_sentiment_mask   = sentiment_mask.view(  -1, seq_len, seq_len)

        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        flat_commonsense_mask = flat_commonsense_mask.view(-1, 1, seq_len, seq_len)
        flat_dependency_mask  = flat_dependency_mask.view( -1, 1, seq_len, seq_len)
        flat_entity_mask      = flat_entity_mask.view(     -1, 1, seq_len, seq_len)
        flat_sentiment_mask   = flat_sentiment_mask.view(  -1, 1, seq_len, seq_len)

        # 不同的head对应不同的mask矩阵
        weight = self.softmax(self.weight)

        # weight: 1 * n_head * 4 * 1 * 1
        weight = weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # flat_mask: (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = torch.cat(
            [flat_commonsense_mask, flat_dependency_mask, flat_entity_mask, flat_sentiment_mask], dim=1)

        # flat_mask: (batch_size * choice_num) * n_head * 4 * seq_len * seq_len
        flat_mask = flat_mask.unsqueeze(1).repeat(1, 12, 1, 1, 1)
        flat_mask = torch.mul(weight.float(), flat_mask.float())

        # slf_attn_mask: (batch_size * choice_num) * n_head * seq_len * seq_len
        slf_attn_mask = torch.sum(flat_mask, dim=2)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num)      * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * n_head * seq_len * seq_len
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        non_pad_mask = non_pad_mask.float()

        # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(-1, seq_len, seq_len)

        # debug: 绝对不能加
        # slf_attn_mask = slf_attn_mask.byte()

        tran_attn = self.transformer_mrc(enc_output=roberta_attn,
                                         non_pad_mask=non_pad_mask,
                                         slf_attn_mask=slf_attn_mask)

        # Debug: pooled_output分类器并非取自sequence_output[:, 0]
        # sequence_output = bert_attn + tran_attn
        # pooled_output = sequence_output[:, 0]

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


class RobertaForMultipleChoice_Fusion_Layer(BertPreTrainedModel):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fusion layer".format(time.ctime(time.time())))
        super(RobertaForMultipleChoice_Fusion_Layer, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder_layer(n_layers=3, n_head=16, d_k=64, d_v=64,
                                                   d_model=1024, d_inner=4096, dropout=0.1)

        self.pooler = BertPooler(config)

        self.weight = nn.Parameter(torch.randn(3, 4))
        self.softmax = nn.Softmax(dim=-1)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,

                attention_mask=None,
                commonsense_mask=None,
                dependency_mask=None,
                entity_mask=None,
                sentiment_mask=None,

                labels=None):

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        entity_mask = entity_mask.unsqueeze(2).expand(batch_size, num_choices, seq_len, seq_len)

        flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids  = None

        flat_attention_mask   = attention_mask.view(  -1, seq_len)
        flat_commonsense_mask = commonsense_mask.view(-1, seq_len, seq_len)
        flat_dependency_mask  = dependency_mask.view( -1, seq_len, seq_len)
        flat_entity_mask      = entity_mask.view(     -1, seq_len, seq_len)
        flat_sentiment_mask   = sentiment_mask.view(  -1, seq_len, seq_len)

        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        flat_attention_mask   = flat_attention_mask.view(  -1, 1, seq_len)
        flat_commonsense_mask = flat_commonsense_mask.view(-1, 1, seq_len, seq_len)
        flat_dependency_mask  = flat_dependency_mask.view( -1, 1, seq_len, seq_len)
        flat_entity_mask      = flat_entity_mask.view(     -1, 1, seq_len, seq_len)
        flat_sentiment_mask   = flat_sentiment_mask.view(  -1, 1, seq_len, seq_len)

        # 不同的layer对应不同的mask矩阵
        # weight: 3 * 4
        weight = self.softmax(self.weight)

        # weight: 3 * 1 * 4 * 1 * 1
        weight = weight.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # flat_mask: (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = torch.cat(
            [flat_commonsense_mask, flat_dependency_mask, flat_entity_mask, flat_sentiment_mask], dim=1)

        # flat_mask: 3 * (batch_size * choice_num) * 4 * seq_len * seq_len
        flat_mask = flat_mask.unsqueeze(0).repeat(3, 1, 1, 1, 1)

        flat_mask = torch.mul(weight.float(), flat_mask.float())

        # slf_attn_mask: 3 * (batch_size * choice_num) * seq_len * seq_len
        slf_attn_mask = torch.sum(flat_mask, dim=2)

        # slf_attn_mask: 3 * (batch_size * choice_num * n_head) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.repeat(1, 12, 1, 1)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num) * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        non_pad_mask = non_pad_mask.float()

        # slf_attn_mask: 3 * (batch_size * choice_num) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(3, -1, seq_len, seq_len)

        # debug: 绝对不能加
        # slf_attn_mask = slf_attn_mask.byte()

        tran_attn = self.transformer_mrc(enc_output=roberta_attn,
                                         non_pad_mask=non_pad_mask,
                                         slf_attn_mask=slf_attn_mask)

        # Debug: pooled_output分类器并非取自sequence_output[:, 0]
        # sequence_output = bert_attn + tran_attn
        # pooled_output = sequence_output[:, 0]

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
