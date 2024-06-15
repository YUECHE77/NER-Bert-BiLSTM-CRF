from .utils_Generic import evaluate


def eval_conll2003(data_loader, model, device, categories):
    """Evaluate the model performance during training"""
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


def eval_cluener(data_loader, model, device, categories):
    """Very similar to the evaluation function above"""
    target_labels = []
    pred_labels = []

    for data in data_loader:
        tokenized_inputs, targets = data
        tokenized_inputs, targets = tokenized_inputs.to(device), targets.to(device)

        pred = model.predict(tokenized_inputs)  # shape: (batch_size, seq_len)

        for target, prediction in zip(targets, pred):
            for idx, i in enumerate(prediction):
                if target[idx] != -100:
                    target_labels.append(categories[target[idx].item()])
                    pred_labels.append(categories[i])

    precision, recall, f1 = evaluate(real_label=target_labels, predict_label=pred_labels)

    return precision, recall, f1
