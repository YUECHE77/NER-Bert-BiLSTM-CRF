import torch

from transformers import BertTokenizerFast

import utils.utils_cluener as C
import utils.utils_Generic as G

from nets.Bert_BiLSTM_CRF_Combined import CombinedNER


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   prompt: The sentence you want to test
    # ----------------------------------------------------#
    prompt = ''
    # ----------------------------------------------------#
    #   data_path:      Path to the training set
    #   test_path:      Path to the test set

    #   pretrained_model_name:  The Bert model
    #   model_path:             Your trained model
    #   use_bilstm:             Did you contain BiLSTM while training
    # ----------------------------------------------------#
    data_path = 'dataset/cluener/train.json'
    test_path = 'dataset/cluener/dev.json'

    pretrained_model_name = 'All_Bert_Pretrained_Models/bert-base-chinese'
    model_path = 'logs/Bert_BiLSTM_CRF_f1_782.pth'
    use_bilstm = True
    # ----------------------------------------------------#
    #   Get the label list and entities categories
    # ----------------------------------------------------#
    label_list, categories = C.get_labellist_and_categories(data_path, test_path)
    # ----------------------------------------------------#
    #   Load the model/tokenizer and put it on GPU
    # ----------------------------------------------------#
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CombinedNER(num_class=len(label_list), bert_type=pretrained_model_name, need_rnn=use_bilstm)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    # ----------------------------------------------------#
    #   Start prediction
    # ----------------------------------------------------#
    entities = G.predict(prompt, model, tokenizer, device, categories, use_crf=True, english=False)

    for entity, name in entities:
        print(f'Entity: {entity}, Type: {name}')
