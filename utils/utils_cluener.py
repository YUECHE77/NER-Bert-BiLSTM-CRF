import json
import numpy as np

from sklearn.model_selection import train_test_split


def read_json(file_path, all_names=set()):
    """Load data from the json file"""
    all_data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # line example:
            #    {"text": "生生不息CSOL生化狂潮让你填弹狂扫", "label": {"game": {"CSOL": [[4, 7]]}}}
            #    {"text": "而且在7个客场打入14球进攻比主场更凶猛，热奥瓦尼、马龙·", "label": {"name": {"热奥瓦尼": [[21, 24]], "马龙": [[26, 27]]}}}
            #    {"text": "□记者姚克勤通讯员李鸿光", "label": {"name": {"姚克勤": [[3, 5]], "李鸿光": [[9, 11]]}, "position": {"记者": [[1, 2]], "通讯员": [[6, 8]]}}}
            line = json.loads(line.strip())  # load the json file -> now is a dictionary

            sentence = line['text']
            sentence = list(sentence)  # Separate each charactor -> become a list -> [A, A, ..., A]

            # "name": {"姚克勤": [[3, 5]], "李鸿光": [[9, 11]]}, "position": {"记者": [[1, 2]], "通讯员": [[6, 8]]}}
            labels = line.get('label', None)

            ner_labels = ['O'] * len(sentence)

            if labels is not None:
                for key, value in labels.items():
                    # key: "name"
                    # value: {"姚克勤": [[3, 5]], "李鸿光": [[9, 11]]} -> value is a dict
                    all_names.add(key)  # Intentionally not adding "I-key"
                    for entity_name, entity_pos in value.items():
                        # entity_name: "姚克勤"
                        # entity_pos: [[3, 5]]
                        for start_pos, end_pos in entity_pos:
                            # start_pos: 3
                            # end_pos: 5
                            assert ''.join(
                                sentence[start_pos:end_pos + 1]) == entity_name, 'The entity position is not correct!'

                            if start_pos == end_pos:
                                ner_labels[start_pos] = 'B-' + key
                            else:
                                ner_labels[start_pos] = 'B-' + key
                                ner_labels[start_pos + 1:end_pos + 1] = ['I-' + key] * (len(entity_name) - 1)

            # all_data -> [([..., ..., ...], [..., ..., ...]), ([..., ..., ...], [..., ..., ...]), ..., ([..., ..., ...], [..., ..., ...])]
            all_data.append((sentence, ner_labels))

    return all_data, all_names


def process_all_names(all_names):
    """Get label list and categories"""
    all_names = list(all_names)
    all_names.sort()  # Sort the entities names -> set is VERY unstable

    label_list = ['O']
    categories = {}

    for name in all_names:
        B = 'B-' + name
        I = 'I-' + name

        label_list += [B, I]

    for idx, label in enumerate(label_list):
        categories[label] = idx
        categories[idx] = label

    return label_list, categories


def split_data(all_data, val_ratio):
    """Split the training data into training set and validation set"""
    train_data, val_data = train_test_split(all_data, test_size=val_ratio, random_state=42)
    return train_data, val_data


def get_train_test_val(data_path='dataset/cluener/train.json', test_path='dataset/cluener/dev.json', val_ratio=0.15):
    """
    Download the cluener dataset, use this function to get training/test/validation data
    Also return the label_list and categories
    They are in the format: [([list1], [list2]), ([list3], [list4]), ..., ([listN], [listN+1])]

    Important: The test.json in cluener dataset is useless -> it doesn't give us labels

    :param data_path: The path to train.json
    :param test_path: The path to dev.json
    :param val_ratio: Ratio of validation set
    :return: train_data, test_data, val_data
    """
    all_names = set()

    # all_data = training data + validation data
    # The following three lines of codes generate training, test, and validation data
    all_data, all_names = read_json(data_path, all_names)
    test_data, all_names = read_json(test_path, all_names)
    train_data, val_data = split_data(all_data, val_ratio=val_ratio)

    label_list, categories = process_all_names(all_names)

    return train_data, test_data, val_data, label_list, categories


def check_data(data, num=1):
    """Check training/test/validation data"""
    length = len(data)
    indices = np.random.choice(length, size=num, replace=False)

    for index in indices:
        sample = data[index]
        sentence = sample[0]
        labels = sample[1]
        print(' '.join(sentence))
        print()
        print(' '.join(labels))
        print('-' * 50)


def get_labellist_and_categories(data_path='dataset/cluener/train.json', test_path='dataset/cluener/dev.json'):
    """Just for inference"""
    all_names = set()

    def helper_func(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())

                labels = line.get('label', None)

                if labels is not None:
                    for key, _ in labels.items():
                        all_names.add(key)

    helper_func(data_path)
    helper_func(test_path)

    label_list, categories = process_all_names(all_names)

    return label_list, categories
