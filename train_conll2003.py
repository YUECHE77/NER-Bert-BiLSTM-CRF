import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizerFast

import utils.utils_conll2003 as E
import utils.utils_Generic as G
from utils.utils_training_testing import eval_conll2003
from nets.Bert_Only import BertNER


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   save_path:      Path to save you json file
    #   download_path:  Path to conll2003 dataset
    #   if_downloaded:  Have downloaded the conll2003 dataset
    # ----------------------------------------------------#
    save_path = 'dataset/conll2023.jsonl'
    download_path = 'dataset/conll2003_NER'
    if_downloaded = True
    # ----------------------------------------------------#
    #   Training parameters
    #   epoch_num       Epoch number
    #   batch_size      Batch size
    #   lr              Learning rate
    # ----------------------------------------------------#
    epoch_num = 1
    batch_size = 2
    lr = 2e-5
    # ----------------------------------------------------#
    #   Download the dataset
    # ----------------------------------------------------#
    label_list, categories = E.get_json_data(save_path=save_path, if_downloaded=if_downloaded,
                                             download_path=download_path)
    # ----------------------------------------------------#
    #   Read in the data and get the dataloaders
    #   pretrained_model_name: Which pretrained bert model to use
    # ----------------------------------------------------#
    train_data, test_data = E.load_json(save_path)

    pretrained_model_name = 'All_Bert_Pretrained_Models/bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    train_loader = G.get_dataloader(train_data, tokenizer, categories=categories, mode='Train')
    val_loader = G.get_dataloader(test_data, tokenizer, categories=categories, mode='Val')
    # ----------------------------------------------------#
    #   Get the model and put it on GPU
    # ----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The device you are using is:', device)

    model = BertNER(len(label_list), bert_model_type=pretrained_model_name)
    model.to(device)
    # ----------------------------------------------------#
    #   Optimizer and Loss function
    # ----------------------------------------------------#
    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    # ----------------------------------------------------#
    #   Start training
    # ----------------------------------------------------#
    for epoch in range(epoch_num):
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for data in train_loader:
                model.train()

                tokenized_inputs, targets = data
                tokenized_inputs, targets = tokenized_inputs.to(device), targets.to(device)
                targets = targets.view(-1)

                outputs = model(tokenized_inputs)
                loss = loss_func(outputs, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        with torch.no_grad():
            model.eval()

            precision, recall, f1 = eval_conll2003(val_loader, model, device, categories)

            print(f'Epoch: {epoch + 1:02d}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.3f}')

            if (epoch + 1) % 5 == 0:
                sub_path = int(f1 * 1000)
                save_path = f'logs/model_f1_{sub_path}.pth'
                torch.save(model.state_dict(), save_path)

    print('Finished Training')
