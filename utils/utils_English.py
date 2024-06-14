import json
from datasets import load_dataset, load_from_disk


def download_conll2023(save_path):
    dataset = load_dataset('conll2003')
    dataset.save_to_disk(save_path)
    print('The dataset is successfully downloaded to:', save_path)


def get_json_data(save_path='dataset/conll2003.jsonl', if_downloaded=False, download_path='dataset/conll2003_NER'):
    """Download(or load) the conll2023_NER dataset, and save the Json file. Return the entity names and dictionary"""
    if not if_downloaded:
        download_conll2023(download_path)

    dataset = load_from_disk(download_path)

    dataset_dict = {}  # Store the data
    categories = {}  # index to label and label to index

    # ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    label_list = dataset['train'].features['ner_tags'].feature.names

    for idx, label in enumerate(label_list):
        categories[label] = idx
        categories[idx] = label

    def tokenize_and_align_labels(examples):
        """
        Process the dataset
            examples['tokens'] -> Sentence
            examples['ner_tags'] -> Corresponding tags
            Both are in the form of list
        """
        for id_, tokens, ner_tags in zip(examples['id'], examples['tokens'], examples['ner_tags']):
            ner_labels = [label_list[i] for i in ner_tags]  # Tags -> From numbers to strings

            # That's the format of Json file(dataset):
            # {
            #   id: {'sents': The sentence, 'ner_labels': The corresponding labels}
            # }
            dataset_dict[id_] = {'sents': ' '.join(tokens), 'ner_labels': ' '.join(ner_labels)}

    datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

    with open(save_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(dataset_dict, indent=4))

    print('The Json file is successfully saved in:', save_path)

    return label_list, categories


def check_json(num_sents=3, save_path='dataset/conll2003.jsonl'):
    with open(save_path, 'r', encoding='utf8') as file:
        data = json.load(file)

    for i, (key, value) in enumerate(data.items()):
        # key -> id
        # value -> {'sents': The sentence, 'ner_labels': The corresponding labels}
        if i < num_sents:  # Only print the first "num_sents" sentences
            print(f"ID: {key}")
            print(f"Sentence: {value['sents']}")
            print(f"NER Labels: {value['ner_labels']}")
            print()
        else:
            break


def load_json(data_path, train_ratio=0.8):
    all_data = []

    with open(data_path, 'r', encoding='utf8') as file:
        data = json.load(file)

    for key, value in data.items():
        # key -> id
        # value -> {'sents': The sentence, 'ner_labels': The corresponding labels}
        sent = value['sents'].split(' ')
        ner_labels = value['ner_labels'].split(' ')

        assert len(sent) == len(ner_labels), "for every sentence, its ner_tag need to have the same length!"

        # That's the format of training/validation/test data:
        # Every data in a tuple -> (Sentence, NER_tag) -> Both in list
        # For example:
        # Sentence: ['SOCCER', '-', 'JAPAN', 'GET', 'LUCKY', 'WIN', ',', 'CHINA', 'IN', 'SURPRISE', 'DEFEAT', '.']
        # NER_tag: ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O']
        all_data.append((sent, ner_labels))

    length = len(all_data)
    train_len = int(length * train_ratio)

    train_data = all_data[:train_len]
    test_data = all_data[train_len:]

    return train_data, test_data



