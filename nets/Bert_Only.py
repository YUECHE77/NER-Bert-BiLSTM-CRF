import torch.nn as nn
from transformers import BertModel


class BertNER(nn.Module):
    def __init__(self, num_class, bert_model_type='bert-base-uncased'):
        super(BertNER, self).__init__()

        # The default value of BertModel's output is 768
        # return_dict=True -> BertModel outputs in dictionary -> can be used as: bert_output['last_hidden_state']
        self.bert = BertModel.from_pretrained(bert_model_type, return_dict=True)

        self.classifier = nn.Linear(768, num_class)

    def forward(self, inputs):
        # input_ids :tensor，shape=batch_size * max_len -> max_len: Length of the longest sentence in current batch
        # input_tyi :tensor，identifiers of two sentences
        # input_attn_mask :tensor，contains only 0 and 1 -> ignore the paddings
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']

        # Two parts of Bert outputs：
        #     1. last_hidden_state: Value of the last hidden layer -> shape: (batch_size, sequence_length, hidden_size)
        #     2. pooler_output: Outputs of [CLS], used in classification task -> shape: (batch_size, hidden_size)
        outputs = self.bert(input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)

        # We need last_hidden_state，instead of pooler_output
        # Because we are classifying tokens(NER)，instead of classifying sentences(Sentiment analysis)
        last_hidden_state = outputs.last_hidden_state

        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, num_class)
        class_results = self.classifier(last_hidden_state)

        batch_size, seq_len, ner_class_num = class_results.shape
        class_results = class_results.view(batch_size * seq_len, ner_class_num)

        return class_results
