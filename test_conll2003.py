import torch

from transformers import BertTokenizerFast

import utils.utils_conll2003 as E
import utils.utils_Generic as G
from nets.Bert_Only import BertNER

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   prompt: The sentence you want to test
    # ----------------------------------------------------#
    prompt = ''
    # ----------------------------------------------------#
    #   save_path:      Path to save you json file
    #   download_path:  Path to conll2003 dataset
    #   if_downloaded:  Have downloaded the conll2003 dataset

    #   pretrained_model_name:  The Bert model
    #   model_path:             Your trained model
    # ----------------------------------------------------#
    download_path = 'dataset/conll2003_NER'
    if_downloaded = True

    pretrained_model_name = 'All_Bert_Pretrained_Models/bert-base-uncased'
    model_path = 'logs/Bert_only_f1_90.pth'
    # ----------------------------------------------------#
    #   Get the label list and entities categories
    # ----------------------------------------------------#
    label_list, categories = E.labellist_and_categories(if_downloaded=if_downloaded, download_path=download_path)
    # ----------------------------------------------------#
    #   Load the model/tokenizer and put it on GPU
    # ----------------------------------------------------#
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BertNER(len(label_list), bert_model_type=pretrained_model_name)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    # ----------------------------------------------------#
    #   Start prediction
    # ----------------------------------------------------#
    entities = G.predict(prompt, model, tokenizer, device, categories, use_crf=False, english=True)

    print('\nStart recognize: \n')

    for entity, name in entities:
        print(f'Entity: {entity}, Type: {name}')

    print('\nFinished recognize \n')
