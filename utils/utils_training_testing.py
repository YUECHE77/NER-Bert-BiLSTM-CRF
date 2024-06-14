import torch

from .utils_generic import evaluate


def eval_conll2003(data_loader, model, device, categories):
    target_labels = []
    pred_labels = []

    for data in data_loader:
        tokenized_inputs, targets = data  # targets shape: (batch_size, seq_len)
        tokenized_inputs, targets = tokenized_inputs.to(device), targets.to(device)
        targets = targets.view(-1)  # shape: (batch_size * seq_len, )

        outputs = model(tokenized_inputs)
        pred = outputs.argmax(dim=-1)  # shape: (batch_size * seq_len, )

        targets_list = targets.tolist()
        pred_list = pred.tolist()

        for target, prediction in zip(targets_list, pred_list):
            if target != -100:
                target_labels.append(categories[target])
                pred_labels.append(categories[prediction])

    precision, recall, f1 = evaluate(real_label=target_labels, predict_label=pred_labels)

    return precision, recall, f1


def preprocess(text, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs


def predict(text, model, tokenizer, device, categories):
    inputs = preprocess(text, tokenizer, device)
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.argmax(dim=-1)

    # 将预测结果转换为标签
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predictions = predictions.view(-1).cpu().numpy()
    labels = [categories[pred] for pred in predictions if pred != -100]

    return tokens, labels


def postprocess(tokens, labels):
    results = []
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:  # 忽略特殊字符
            continue
        if token.startswith("##"):
            results[-1][0] += token[2:]
        else:
            results.append([token, label])
    return results
