import torch

from transformers import BertTokenizerFast
from transformers import AdamW

from tqdm import tqdm

import utils.utils_cluener as C
import utils.utils_Generic as G
from utils.utils_training_testing import eval_cluener
from nets.Bert_BiLSTM_CRF_Combined import CombinedNER


if __name__ == '__main__':
    # ----------------------------------------------------#
    #   data_path:      Path to the training set
    #   test_path:      Path to the test set
    #   val_ratio:      Ratio of validation set

    #   pretrained_model_name: Which pretrained bert model to use
    #   use_bilstm:            Use BiLSTM or not

    #   test_performance:      Use test set to test the performance
    # ----------------------------------------------------#
    data_path = 'dataset/cluener/train.json'
    test_path = 'dataset/cluener/dev.json'
    val_ratio = 0.15

    pretrained_model_name = 'All_Bert_Pretrained_Models/bert-base-chinese'
    use_bilstm = True

    test_performance = False
    # ----------------------------------------------------#
    #   Training parameters
    #   epoch_num       Epoch number
    #   batch_size      Batch size

    #   Important:      Theoretically, the learning rate for pretrained bert model,
    #                   and the learning rate for LSTM should be different!!!
    #   lr_bert         Learning rate for bert model
    #   lr_other        Learning rate for other layers
    # ----------------------------------------------------#
    epoch_num = 1
    batch_size = 2
    lr_bert = 1e-5
    lr_other = 1e-4
    # ----------------------------------------------------#
    #   Read in the data and get the dataloaders
    # ----------------------------------------------------#
    train_data, test_data, val_data, label_list, categories = C.get_train_test_val(data_path, test_path, val_ratio)

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    train_loader = G.get_dataloader(train_data, tokenizer, categories, mode='Train')
    test_loader = G.get_dataloader(test_data, tokenizer, categories, mode='Test')
    val_loader = G.get_dataloader(val_data, tokenizer, categories, mode='Val')
    # ----------------------------------------------------#
    #   Get the model and put it on GPU
    # ----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CombinedNER(len(label_list), bert_type=pretrained_model_name, need_rnn=use_bilstm)
    model.to(device)
    # ----------------------------------------------------#
    #   Set up the optimizer
    #   We don't need the loss function here -> CRF has already done this part
    # ----------------------------------------------------#
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': lr_bert},
        {'params': model.rnn.parameters(), 'lr': lr_other, 'weight_decay': 1e-4} if model.need_rnn else {'params': []},
        {'params': model.classifier.parameters(), 'lr': lr_other},
        {'params': model.crf.parameters(), 'lr': lr_other}
    ])
    # ----------------------------------------------------#
    #   Start training
    # ----------------------------------------------------#
    print('\nStart training!!!\n')
    for epoch in range(epoch_num):
        total_batches = len(train_loader)

        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for data in train_loader:
                model.train()

                tokenized_inputs, targets = data

                tokenized_inputs, targets = tokenized_inputs.to(device), targets.to(device)

                loss = model(tokenized_inputs, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        with torch.no_grad():
            model.eval()

            precision, recall, f1 = eval_cluener(val_loader, model, device, categories)

            print(f'Epoch: {epoch + 1:02d}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.3f}')

            if (epoch + 1) % 5 == 0:
                sub_path = int(f1 * 1000)
                save_path = f'logs/model_f1_{sub_path}.pth'
                torch.save(model.state_dict(), save_path)

    print('\nFinished Training!!!\n')
    # ----------------------------------------------------#
    #   If you want to test the model performance after training
    # ----------------------------------------------------#
    if test_performance:
        with torch.no_grad():
            model.eval()

            precision, recall, f1 = eval_cluener(test_loader, model, device, categories)

            print(f'On the test set:\n Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.3f}')
