import os

import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def train_func(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """
    :param model:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param device:
    :param epoch:
    :param num_epochs:
    :return:
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, total=len(train_loader), desc="Training")
    for batch in pbar:
        pre_inputs = batch["url_pre"].to(device)
        suf_inputs = batch["url_suf"].to(device)
        state_inputs = batch["features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].to(device)
        targets = targets.squeeze()
        optimizer.zero_grad()
        outputs = model(pre_inputs, suf_inputs, state_inputs, mask)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        predicted = (outputs >= 0.5).int().squeeze()
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        aver_accuracy = total_correct / total_samples
        aver_loss = total_loss / total_samples
        pbar.set_description(
            f"Epoch [{epoch + 1}/{num_epochs}] aver Accuracy: {aver_accuracy:.4f} aver loss: {aver_loss:.8f}")
    accuracy = total_correct / total_samples
    average_loss = total_loss / total_samples
    return average_loss, accuracy


def test_func(model, train_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0
    nums = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        pbar = tqdm(train_loader, total=len(train_loader), desc="Testing")
        for batch in pbar:
            pre_inputs = batch["url_pre"].to(device)
            suf_inputs = batch["url_suf"].to(device)
            state_inputs = batch["features"].to(device)
            mask = batch["mask"].to(device)
            nums += len(batch["label"])
            targets = batch["label"].to(device)
            targets = targets.squeeze()
            test_outputs = model(pre_inputs, suf_inputs, state_inputs, mask)
            test_loss += criterion(test_outputs.squeeze(), targets.float()).item() * targets.size(0)
            test_accuracy += ((test_outputs >= 0.5).int().squeeze() == targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend((test_outputs >= 0.5).int().squeeze().cpu().numpy())
            # Calculate additional evaluation metrics
            f1 = f1_score(all_targets, all_predictions)
            recall = recall_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions)
            pbar.set_description(
                f"test loss:{test_loss / nums:.4f},test Accuracy: {test_accuracy / nums:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
        return test_loss / nums, test_accuracy / nums, f1, recall, precision
