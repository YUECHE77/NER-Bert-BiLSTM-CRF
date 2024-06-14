from .utils_generic import evaluate


def eval_eng(data_loader, model, device, categories):
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

        break

    precision, recall, f1 = evaluate(real_label=target_labels, predict_label=pred_labels)

    return precision, recall, f1
