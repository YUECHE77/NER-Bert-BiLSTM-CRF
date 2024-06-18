import torch
import torch.nn as nn

from transformers import BertModel

from torchcrf import CRF


class CombinedNER(nn.Module):
    def __init__(self, num_class, bert_type='bert-base-chinese', need_rnn=False, rnn_dim=128, drop_rate=0.3):
        super(CombinedNER, self).__init__()

        # return_dict=True -> BertModel outputs in dictionary -> can be used as: bert_output['last_hidden_state']
        self.bert = BertModel.from_pretrained(bert_type, return_dict=True)

        # The default value of BertModel's output is 768
        out_dim = 768

        self.need_rnn = need_rnn
        if self.need_rnn:
            self.rnn = nn.LSTM(out_dim, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(out_dim, num_class)

        self.crf = CRF(num_class, batch_first=True)

    def original_output(self, inputs):
        # input_ids :tensor，shape: (batch_size, max_len) -> max_len: Length of the longest sentence in current batch
        # input_tyi :tensor，identifiers of two sentences
        # input_attn_mask :tensor，contains only 0 and 1 -> ignore the paddings

        # Their dimension are all the same：(Batch_size, Max_sentence_length)
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']

        # Two parts of Bert outputs：
        #     1. last_hidden_state: Value of the last hidden layer -> shape: (batch_size, sequence_length, hidden_size)
        #     2. pooler_output: Outputs of [CLS], used in classification task -> shape: (batch_size, hidden_size)
        outputs = self.bert(input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)

        # We need last_hidden_state，instead of pooler_output
        # Because we are classifying tokens(NER)，instead of classifying sentences(Sentiment analysis)
        sequence_output = outputs.last_hidden_state

        if self.need_rnn:
            sequence_output, _ = self.rnn(sequence_output)

        sequence_output = self.dropout(sequence_output)

        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, num_class)
        temp_results = self.classifier(sequence_output)

        return temp_results

    def forward(self, inputs, targets):
        # Replace -100 with 0 -> very reasonable
        # I think the targets_mask and attention_mask are exactly the same
        targets_mask = (targets != -100).byte()  # Contains only 0 and 1
        targets = torch.where(targets == -100, torch.zeros_like(targets), targets)

        # Make sure the mask values for the first time stap are 1's -> required for CRF
        targets_mask[:, 0] = 1

        temp_results = self.original_output(inputs)

        # Use the targets and targets_mask above to compute the loss
        # Shape of temp_results: (batch_size, seq_len, num_labels) -> Shape of targets should be: (batch_size, seq_len)
        # Just like CrossEntropy
        loss = -1 * self.crf(temp_results, targets, mask=targets_mask)

        return loss

    def predict(self, inputs):
        input_attn_mask = inputs['attention_mask']

        temp_results = self.original_output(inputs)
        pred = self.crf.decode(temp_results, input_attn_mask.byte())

        return pred
