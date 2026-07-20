import torch


def accuracy(output, target):
    """
    Compute classification accuracy (%).
    """
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)

    return 100.0 * correct / total


def per_class_accuracy(output, target, num_classes):
    """
    Compute accuracy for each class.
    """
    pred = torch.argmax(output, dim=1)

    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for i in range(target.size(0)):
        label = target[i]
        class_total[label] += 1

        if pred[i] == label:
            class_correct[label] += 1

    acc = []

    for i in range(num_classes):
        if class_total[i] == 0:
            acc.append(0.0)
        else:
            acc.append(
                100.0 * class_correct[i].item() / class_total[i].item()
            )

    return acc


def confusion_matrix(output, target, num_classes):
    """
    Compute confusion matrix.
    """
    pred = torch.argmax(output, dim=1)

    cm = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64
    )

    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1

    return cm