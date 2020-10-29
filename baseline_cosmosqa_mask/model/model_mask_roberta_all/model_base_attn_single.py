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

from baseline_cosmosqa_mask.model.model_transformer.Layers import EncoderLayer


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


class RobertaForMultipleChoice_Fusion_Layer(BertPreTrainedModel):
    # Commonsense_Mask + Dependency_Mask + Entity_Mask + Sentiment_Mask
    def __init__(self, config):
        print("[TIME] --- time: {} ---, init model fusion layer".format(time.ctime(time.time())))
        super(RobertaForMultipleChoice_Fusion_Layer, self).__init__(config)

        self.n_head = int(config.num_attention_heads)
        self.n_layer = 3
        self.hidden_size = int(config.hidden_size)
        self.layer_size  = int(self.hidden_size / self.n_layer)

        self.roberta = RobertaModel(config)
        self.transformer_mrc = Trans_Encoder(n_layers=3,
                                             n_head=12,
                                             d_k=64,
                                             d_v=64,
                                             d_model=768,
                                             d_inner=4096,
                                             dropout=0.1)

        self.pooler = BertPooler(config)

        self.bn         = torch.nn.BatchNorm1d(num_features=config.hidden_size)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
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
        # flat_input_ids:        [batch_size * choice_num, seq_len]
        # flat_attention_mask:   [batch_size * choice_num, seq_len]
        # flat_commonsense_mask: [batch_size * choice_num, seq_len, seq_len]
        # flat_dependency_mask:  [batch_size * choice_num, seq_len, seq_len]
        # flat_entity_mask:      [batch_size * choice_num, seq_len, seq_len]
        # flat_sentiment_mask:   [batch_size * choice_num, seq_len, seq_len]

        flat_position_ids    = None
        flat_head_mask       = None

        # bert_attn:     [batch_size * choice_num, seq_len, hidden]
        # pooled_output: [batch_size * choice_num, hidden]
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
        # flat_input_ids:        [batch_size * choice_num, 1, seq_len]
        # flat_attention_mask:   [batch_size * choice_num, 1, seq_len]
        # flat_commonsense_mask: [batch_size * choice_num, 1, seq_len, seq_len]
        # flat_dependency_mask:  [batch_size * choice_num, 1, seq_len, seq_len]
        # flat_entity_mask:      [batch_size * choice_num, 1, seq_len, seq_len]
        # flat_sentiment_mask:   [batch_size * choice_num, 1, seq_len, seq_len]

        # slf_attn_mask: (batch_size * choice_num) * n_head * seq_len * seq_len
        # slf_attn_mask = flat_commonsense_mask.unsqueeze(1).repeat(1, 1, self.n_head, 1, 1)
        # slf_attn_mask =  flat_dependency_mask.unsqueeze(1).repeat(1, 1, self.n_head, 1, 1)
        # slf_attn_mask =      flat_entity_mask.unsqueeze(1).repeat(1, 1, self.n_head, 1, 1)
        slf_attn_mask =   flat_sentiment_mask.unsqueeze(1).repeat(1, 1, self.n_head, 1, 1)

        # slf_attn_mask: (batch_size * choice_num * n_head) * seq_len * seq_len
        slf_attn_mask = slf_attn_mask.view(-1, seq_len, seq_len).float()

        # -- Prepare masks
        # non_pad_mask:  (batch_size * choice_num) * seq_len * 1
        # slf_attn_mask: (batch_size * choice_num) * seq_len * seq_len
        non_pad_mask = get_non_pad_mask(flat_input_ids)
        if   torch.get_default_dtype() == torch.float32: non_pad_mask = non_pad_mask.float()
        elif torch.get_default_dtype() == torch.float64: non_pad_mask = non_pad_mask.double()
        else: raise TypeError

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
