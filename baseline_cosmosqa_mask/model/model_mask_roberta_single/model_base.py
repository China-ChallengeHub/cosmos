import sys
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

sys.path.append("../../../")
sys.path.append("../../../baseline_cosmosqa_mask")
sys.path.append("../../../baseline_cosmosqa_mask/run_roberta_transformer")
sys.path.append("../../../baseline_cosmosqa_mask/run_roberta_transformer/model_transformer")

from baseline_cosmosqa_mask.model.modeling_roberta import RobertaModel
from baseline_cosmosqa_mask.model.modeling_roberta import BertPreTrainedModel

from baseline_cosmosqa_mask.model.model_transformer.Layers import EncoderLayer
# from baseline_cosmosqa_mask.model_transformer.Models import get_non_pad_mask
# from baseline_cosmosqa_mask.model_transformer.Models import get_attn_key_pad_mask


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


class RobertaForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=12, d_k=64, d_v=64,
                                             d_model=768, d_inner=3072, dropout=0.1)

        self.n_head = 12
        self.pooler = BertPooler(config)

        self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                prior_mask=None,
                labels=None):

        # input_ids:        batch_size * choice_num * seq_len
        # token_type_ids:   batch_size * choice_num * seq_len
        # attention_mask:   batch_size * choice_num * seq_len
        # commonsense_mask: batch_size * choice_num * seq_len * seq_len

        # flat_input_ids:        (batch_size * choice_num) * seq_len
        # flat_token_type_ids:   (batch_size * choice_num) * seq_len
        # flat_attention_mask:   (batch_size * choice_num) * seq_len
        # flat_commonsense_mask: (batch_size * choice_num) * seq_len * seq_len

        batch_size  = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        seq_len     = input_ids.shape[2]

        flat_input_ids        = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids   = None
        flat_attention_mask   = attention_mask.view(-1, attention_mask.size(-1))
        flat_prior_mask       = prior_mask.view(-1, seq_len, seq_len)
        flat_position_ids     = None
        flat_head_mask        = None

        # roberta_attn:  (batch_size * choice_num) * seq_len * hidden
        # pooled_output: (batch_size * choice_num) * hidden
        roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                   token_type_ids=flat_token_type_ids,
                                                   attention_mask=flat_attention_mask,
                                                   position_ids=flat_position_ids,
                                                   head_mask=flat_head_mask)

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num) * seq_len *    1
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        non_pad_mask = non_pad_mask.float()

        # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
        slf_attn_mask = flat_prior_mask

        # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)

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


# class RobertaForMultipleChoice_Commonsense_Mask(BertPreTrainedModel):
#     def __init__(self, config):
#         super(RobertaForMultipleChoice_Commonsense_Mask, self).__init__(config)
#
#         self.roberta = RobertaModel(config)
#         self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=12, d_k=64, d_v=64,
#                                              d_model=768, d_inner=3072, dropout=0.1)
#
#         self.n_head = 12
#         self.pooler = BertPooler(config)
#
#         self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         self.init_weights()
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 attention_mask=None,
#                 commonsense_mask=None,
#                 labels=None):
#
#         # input_ids:        batch_size * choice_num * seq_len
#         # token_type_ids:   batch_size * choice_num * seq_len
#         # attention_mask:   batch_size * choice_num * seq_len
#         # commonsense_mask: batch_size * choice_num * seq_len * seq_len
#
#         # flat_input_ids:        (batch_size * choice_num) * seq_len
#         # flat_token_type_ids:   (batch_size * choice_num) * seq_len
#         # flat_attention_mask:   (batch_size * choice_num) * seq_len
#         # flat_commonsense_mask: (batch_size * choice_num) * seq_len * seq_len
#
#         batch_size  = input_ids.shape[0]
#         num_choices = input_ids.shape[1]
#         seq_len     = input_ids.shape[2]
#
#         flat_input_ids        = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids   = None
#         flat_attention_mask   = attention_mask.view(-1, attention_mask.size(-1))
#         flat_commonsense_mask = commonsense_mask.view(-1, seq_len, seq_len)
#         flat_position_ids     = None
#         flat_head_mask        = None
#
#         # roberta_attn:  (batch_size * choice_num) * seq_len * hidden
#         # pooled_output: (batch_size * choice_num) * hidden
#         roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
#                                                    token_type_ids=flat_token_type_ids,
#                                                    attention_mask=flat_attention_mask,
#                                                    position_ids=flat_position_ids,
#                                                    head_mask=flat_head_mask)
#
#         # -- Prepare masks
#         # non_pad_mask:  (batch_size * choice_num) * seq_len *    1
#         non_pad_mask = get_non_pad_mask(flat_input_ids)
#         non_pad_mask = non_pad_mask.float()
#
#         # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
#         slf_attn_mask = flat_commonsense_mask
#
#         # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
#         slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
#
#         slf_attn_mask = slf_attn_mask.byte()
#         tran_attn = self.transformer_mrc(enc_output=roberta_attn,
#                                          non_pad_mask=non_pad_mask,
#                                          slf_attn_mask=slf_attn_mask)
#
#         # Debug: pooled_output分类器并非取自sequence_output[:, 0]
#         # sequence_output = bert_attn + tran_attn
#         # pooled_output = sequence_output[:, 0]
#
#         sequence_output = roberta_attn + tran_attn
#         pooled_output = self.pooler(sequence_output)
#
#         pooled_output = self.bn(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#
#         # logits:          (batch_size * num_choices)
#         # reshaped_logits: (batch_size,  num_choices)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             return loss, reshaped_logits
#         else:
#             return reshaped_logits
#
#
# class RobertaForMultipleChoice_Dependency_Mask(BertPreTrainedModel):
#     def __init__(self, config):
#         super(RobertaForMultipleChoice_Dependency_Mask, self).__init__(config)
#
#         self.roberta = RobertaModel(config)
#         self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=12, d_k=64, d_v=64,
#                                              d_model=768, d_inner=3072, dropout=0.1)
#
#         self.pooler = BertPooler(config)
#
#         self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         self.init_weights()
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 attention_mask=None,
#                 dependency_mask=None,
#                 labels=None):
#
#         # input_ids:       batch_size * choice_num * seq_len
#         # token_type_ids:  batch_size * choice_num * seq_len
#         # attention_mask:  batch_size * choice_num * seq_len
#         # dependency_mask: batch_size * choice_num * seq_len * seq_len
#
#         # flat_input_ids:       (batch_size * choice_num) * seq_len
#         # flat_token_type_ids:  (batch_size * choice_num) * seq_len
#         # flat_attention_mask:  (batch_size * choice_num) * seq_len
#         # flat_dependency_mask: (batch_size * choice_num) * seq_len * seq_len
#
#         batch_size  = input_ids.shape[0]
#         num_choices = input_ids.shape[1]
#         seq_len     = input_ids.shape[2]
#
#         flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids  = None
#         flat_attention_mask  = attention_mask.view(-1, attention_mask.size(-1))
#         flat_dependency_mask = dependency_mask.view(-1, seq_len, seq_len)
#         flat_position_ids    = None
#         flat_head_mask       = None
#
#         # roberta_attn:  (batch_size * choice_num) * seq_len * hidden
#         # pooled_output: (batch_size * choice_num) * hidden
#         roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
#                                                    token_type_ids=flat_token_type_ids,
#                                                    attention_mask=flat_attention_mask,
#                                                    position_ids=flat_position_ids,
#                                                    head_mask=flat_head_mask)
#
#         # -- Prepare masks
#         # non_pad_mask:  (batch_size * choice_num) * seq_len *    1
#         non_pad_mask = get_non_pad_mask(flat_input_ids)
#         non_pad_mask = non_pad_mask.float()
#
#         # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
#         slf_attn_mask = flat_dependency_mask
#
#         # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
#         slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
#
#         slf_attn_mask = slf_attn_mask.byte()
#         tran_attn = self.transformer_mrc(enc_output=roberta_attn,
#                                          non_pad_mask=non_pad_mask,
#                                          slf_attn_mask=slf_attn_mask)
#
#         sequence_output = roberta_attn + tran_attn
#         pooled_output = self.pooler(sequence_output)
#
#         pooled_output = self.bn(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#
#         # logits:          (batch_size * num_choices)
#         # reshaped_logits: (batch_size,  num_choices)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             return loss, reshaped_logits
#         else:
#             return reshaped_logits
#
#
# class RobertaForMultipleChoice_Entity_Mask(BertPreTrainedModel):
#     def __init__(self, config):
#         super(RobertaForMultipleChoice_Entity_Mask, self).__init__(config)
#
#         self.roberta = RobertaModel(config)
#         self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=12, d_k=64, d_v=64,
#                                              d_model=768, d_inner=3072, dropout=0.1)
#
#         self.pooler = BertPooler(config)
#
#         self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         self.init_weights()
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 attention_mask=None,
#                 entity_mask=None,
#                 labels=None):
#
#         # input_ids:      batch_size * choice_num * seq_len
#         # token_type_ids: batch_size * choice_num * seq_len
#         # attention_mask: batch_size * choice_num * seq_len
#         # entity_mask:    batch_size * choice_num * seq_len
#
#         # flat_input_ids:      (batch_size * choice_num) * seq_len
#         # flat_token_type_ids: (batch_size * choice_num) * seq_len
#         # flat_attention_mask: (batch_size * choice_num) * seq_len
#         # flat_entity_mask:    (batch_size * choice_num) * seq_len
#
#         batch_size  = input_ids.shape[0]
#         num_choices = input_ids.shape[1]
#         seq_len     = input_ids.shape[2]
#
#         flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids  = None
#         flat_attention_mask  = attention_mask.view(-1, attention_mask.size(-1))
#         flat_entity_mask     = entity_mask.view(-1, entity_mask.size(-1))
#         flat_position_ids    = None
#         flat_head_mask       = None
#
#         # bert_attn:     (batch_size * choice_num) * seq_len * hidden
#         # pooled_output: (batch_size * choice_num) * hidden
#         roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
#                                                    token_type_ids=flat_token_type_ids,
#                                                    attention_mask=flat_attention_mask,
#                                                    position_ids=flat_position_ids,
#                                                    head_mask=flat_head_mask)
#
#         # flat_entity_mask: (batch_size * choice_num) *    1    * seq_len
#         # flat_entity_mask: (batch_size * choice_num) * seq_len * seq_len
#         flat_entity_mask = flat_entity_mask.unsqueeze(1)
#         flat_entity_mask = flat_entity_mask.expand(-1, seq_len, seq_len)
#
#         # debug: 在不同的模型中，仅仅是slf_attn_mask不同
#         # debug: slf_attn_mask为None或者不为None时，效果均一样
#
#         # -- Prepare masks
#         # non_pad_mask:  (batch_size * choice_num) * seq_len *    1
#         non_pad_mask = get_non_pad_mask(flat_input_ids)
#         non_pad_mask = non_pad_mask.float()
#
#         # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
#         slf_attn_mask = flat_entity_mask
#
#         # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
#         slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
#
#         slf_attn_mask = slf_attn_mask.byte()
#
#         tran_attn = self.transformer_mrc(enc_output=roberta_attn,
#                                          non_pad_mask=non_pad_mask,
#                                          slf_attn_mask=slf_attn_mask)
#
#         sequence_output = roberta_attn + tran_attn
#         pooled_output = self.pooler(sequence_output)
#
#         pooled_output = self.bn(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#
#         # logits:          (batch_size * num_choices)
#         # reshaped_logits: (batch_size,  num_choices)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             return loss, reshaped_logits
#         else:
#             return reshaped_logits
#
#
# class RobertaForMultipleChoice_Ner_Mask(BertPreTrainedModel):
#     def __init__(self, config):
#         super(RobertaForMultipleChoice_Ner_Mask, self).__init__(config)
#
#         self.roberta = RobertaModel(config)
#         self.transformer_mrc = Trans_Encoder(n_layers=3, n_head=12, d_k=64, d_v=64,
#                                              d_model=768, d_inner=3072, dropout=0.1)
#
#         self.pooler = BertPooler(config)
#
#         self.bn = torch.nn.BatchNorm1d(num_features=config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#         self.init_weights()
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 attention_mask=None,
#                 ner_mask=None,
#                 labels=None):
#
#         # input_ids:      batch_size * choice_num * seq_len
#         # token_type_ids: batch_size * choice_num * seq_len
#         # attention_mask: batch_size * choice_num * seq_len
#         # ner_mask:       batch_size * choice_num * seq_len
#
#         # flat_input_ids:      (batch_size * choice_num) * seq_len
#         # flat_token_type_ids: (batch_size * choice_num) * seq_len
#         # flat_attention_mask: (batch_size * choice_num) * seq_len
#         # flat_ner_mask:       (batch_size * choice_num) * seq_len
#
#         batch_size = input_ids.shape[0]
#         num_choices = input_ids.shape[1]
#         seq_len = input_ids.shape[2]
#
#         flat_input_ids       = input_ids.view(-1, input_ids.size(-1))
#         flat_token_type_ids  = None
#         flat_attention_mask  = attention_mask.view(-1, attention_mask.size(-1))
#         flat_ner_mask        = ner_mask.view(-1, ner_mask.size(-1))
#         flat_position_ids    = None
#         flat_head_mask       = None
#
#         # bert_attn:     (batch_size * choice_num) * seq_len * hidden
#         # pooled_output: (batch_size * choice_num) * hidden
#         # bert_attn:     (batch_size * choice_num) * seq_len * hidden
#         # pooled_output: (batch_size * choice_num) * hidden
#         roberta_attn, pooled_output = self.roberta(input_ids=flat_input_ids,
#                                                    token_type_ids=flat_token_type_ids,
#                                                    attention_mask=flat_attention_mask,
#                                                    position_ids=flat_position_ids,
#                                                    head_mask=flat_head_mask)
#
#         # flat_ner_mask: (batch_size * choice_num)  *   1    * seq_len
#         # flat_ner_mask: (batch_size * choice_num) * seq_len * seq_len
#         flat_ner_mask = flat_ner_mask.unsqueeze(1)
#         flat_ner_mask = flat_ner_mask.expand(-1, seq_len, seq_len)
#
#         # -- Prepare masks
#         # non_pad_mask:  (batch_size * choice_num) * seq_len *    1
#         non_pad_mask = get_non_pad_mask(flat_input_ids)
#         non_pad_mask = non_pad_mask.float()
#
#         # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
#         slf_attn_mask = flat_ner_mask
#
#         # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
#         slf_attn_mask = slf_attn_mask.repeat(self.n_head, 1, 1)
#
#         slf_attn_mask = slf_attn_mask.byte()
#         tran_attn = self.transformer_mrc(enc_output=roberta_attn,
#                                          non_pad_mask=non_pad_mask,
#                                          slf_attn_mask=slf_attn_mask)
#
#         sequence_output = roberta_attn + tran_attn
#         pooled_output = self.pooler(sequence_output)
#
#         pooled_output = self.bn(pooled_output)
#         pooled_output = self.dropout(pooled_output)
#
#         # logits:          (batch_size * num_choices)
#         # reshaped_logits: (batch_size,  num_choices)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)
#
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)
#             return loss, reshaped_logits
#         else:
#             return reshaped_logits
