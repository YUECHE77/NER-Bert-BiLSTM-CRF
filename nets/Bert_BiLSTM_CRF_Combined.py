import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from transformers import logging
from transformers import AdamW

from datasets import load_dataset

from torchcrf import CRF


class BertNER(nn.Module):
    def __init__(self, num_class, bert_type='bert-base-chinese', need_rnn=False, rnn_dim=128, drop_rate=0.3):
        super(BertNER, self).__init__()

        # # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(bert_type, return_dict=True)

        # BertModel的最终输出维度默认为768
        out_dim = 768

        self.need_rnn = need_rnn
        if self.need_rnn:
            self.rnn = nn.LSTM(out_dim, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(out_dim, num_class)

        self.crf = CRF(num_class, batch_first=True)

    def original_output(self, inputs):
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，区分两个句子的标识符（通常用于句子对分类任务中）
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        # 这三个的维度是一样的：(Batch_size, Max_sentence_length)
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']

        # Bert outputs 分为两个部分：
        #   last_hidden_state:最后一个隐层的值 -> shape: (batch_size, sequence_length, hidden_size)
        #   pooler_output:对应的是[CLS]的输出,用于分类任务 -> shape: (batch_size, hidden_size)
        outputs = self.bert(input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)

        # 虽然是分类任务，但我们需要的是last_hidden_state，而不是pooler_output
        # 因为我们要对每一个token进行分类（NER），而不是对整个句子进行分类（情感分析）
        sequence_output = outputs.last_hidden_state

        if self.need_rnn:
            sequence_output, _ = self.rnn(sequence_output)

        sequence_output = self.dropout(sequence_output)

        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, num_class)
        temp_results = self.classifier(sequence_output)

        return temp_results

    def forward(self, inputs, targets):
        # 将-100标签替换为0，并在掩码中相应位置标记为0
        # targets_mask和attention_mask的维度完全一致 -> 我认为这俩就是完全一样的
        targets_mask = (targets != -100).byte()  # Contains only 0 and 1
        targets = torch.where(targets == -100, torch.zeros_like(targets), targets)  # Replace -100 with 0 -> very resonable

        # 确保所有序列的第一个时间步的掩码值都为1 -> required for CRF
        targets_mask[:, 0] = 1

        temp_results = self.original_output(inputs)

        # 计算损失时使用修改后的targets和掩码
        # CRF 层期望输入的targets与temp_results的形状一致。
        # temp_results的形状为 (batch_size, seq_len, num_labels)，因此targets的形状应该为 (batch_size, seq_len)
        loss = -1 * self.crf(temp_results, targets, mask=targets_mask)

        return loss

    def predict(self, inputs):
        input_attn_mask = inputs['attention_mask']

        temp_results = self.original_output(inputs)
        pred = self.crf.decode(temp_results, input_attn_mask.byte())

        return pred
