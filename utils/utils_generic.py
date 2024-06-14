import torch
from torch.utils.data import DataLoader
from .NER_Dataset import NerDataset


def get_dataloader(data, tokenizer, categories, batch_size=32, mode='Train'):
    def collate_fn(examples):
        """The call back function"""
        sents, all_labels = [], []

        for sent, ner_label in examples:
            sents.append(sent)  # [[..., ..., ...], [..., ..., ...], [..., ..., ...], ... ...]
            all_labels.append([categories[i] for i in ner_label])  # From strings to numbers

        tokenized_inputs = tokenizer(sents, truncation=True, padding=True, return_offsets_mapping=True,
                                     is_split_into_words=True, max_length=512, return_tensors='pt')

        targets = []

        for idx, label in enumerate(all_labels):
            # label example -> ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']

            label_ids = []

            for word_idx in tokenized_inputs.word_ids(batch_index=idx):
                # word_idx sample: [None, 0, 1, 2, 2, 3, 3, None] -> Numbers represent the original words indices
                # Set the label of special symbols to -100 -> automatically ignored when calculating the loss
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])

            targets.append(label_ids)

        targets = torch.tensor(targets)

        return tokenized_inputs, targets

    if_shuffle = True if mode == 'Train' else False

    dataset = NerDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=if_shuffle)

    return dataloader


def check_one_batch(dataloader, inspection='input_ids'):
    for data in dataloader:
        tokenized_inputs, targets = data
        print(inspection, ':', tokenized_inputs[inspection])
        print(targets)
        break


def split_entity(label_sequence):
    entity_mark = []
    entity_pointer = None
    for index, label in enumerate(label_sequence):
        if label.startswith('B'):
            if entity_pointer is not None:
                entity_mark.append(entity_pointer)
            category = label.split('-')[1]
            entity_pointer = [index, category, label]
        elif label.startswith('I'):
            if entity_pointer is None:
                continue
            if entity_pointer[1] != label.split('-')[1]:
                entity_pointer = None
                continue
            entity_pointer.append(label)
        else:
            if entity_pointer is not None:
                entity_mark.append(entity_pointer)
            entity_pointer = None
    if entity_pointer is not None:
        entity_mark.append(entity_pointer)

    return entity_mark


def evaluate(real_label, predict_label):
    real_entity_mark = split_entity(real_label)
    predict_entity_mark = split_entity(predict_label)

    true_entity_mark = [entity for entity in predict_entity_mark if entity in real_entity_mark]

    real_entity_num = len(real_entity_mark)
    predict_entity_num = len(predict_entity_mark)
    true_entity_num = len(true_entity_mark)

    precision = true_entity_num / predict_entity_num if predict_entity_num > 0 else 0
    recall = true_entity_num / real_entity_num if real_entity_num > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
