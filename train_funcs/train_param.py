import os
from random import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from train_funcs.train_func import train_func, test_func
from load_data import get_dataset
import adabound


def train_1_param(model_func, args, ):
    print("loading data.....")
    acc_ls = []
    result, feature_len = get_dataset(**args["dataset_params"])
    train_loader, val_loader, test_loader = result["train"], result["val"], result["test"]
    print("end loading data....")
    # args["model_params"]["mask_mlp_params"]["features_len"] = feature_len
    model = model_func(args["dataset_params"]["pre_len"],
                       args["dataset_params"]["suf_len"],
                       args["model_params"]["the_max_pool_output_size"],
                       args["model_params"]["tcn_params"],
                       args["model_params"]["mask_atte_params"],
                       args["model_params"]["sc_attention_params"],
                       args["model_params"]["feature_fusion_params"],
                       )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], betas=args["betas"], weight_decay=args["weight_decay"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
    model.to(device)
    criterion.to(device)
    num_epochs = args["num_epochs"]

    writer = None
    current_datetime = datetime.now()
    data_time = current_datetime.strftime("(%Y-%m-%d)-(%H-%M)-%S")

    if args["logs_dir"]:
        writer = SummaryWriter(args["logs_dir"] + "/" + data_time)

    for epoch in range(num_epochs):
        train_epoch_loss, train_epoch_acc = train_func(model, train_loader, criterion, optimizer, device, epoch,
                                                       num_epochs)
        result = test_func(model, test_loader, criterion, device)
        # val_result = test_func(model,val_loader,criterion, device)
        acc_ls.append(result)
        if args["logs_dir"]:
            writer.add_scalar('Loss/train_funcs', train_epoch_loss, epoch)
            writer.add_scalar('Accuracy/train_funcs', train_epoch_acc, epoch)

            writer.add_scalar('Loss/test', result[0], epoch)
            writer.add_scalar('Accuracy/test', result[1], epoch)
            writer.add_scalar('F1 Score', result[2], epoch)
            writer.add_scalar('Recall', result[3], epoch)
            writer.add_scalar('Precision', result[4], epoch)
    if args["logs_dir"]:
        writer.close()
    return acc_ls, data_time, args

def train_2_param(model_func, args, ):
    print("loading data.....")
    acc_ls = []
    result, feature_len = get_dataset(**args["dataset_params"])
    train_loader, val_loader, test_loader = result["train"], result["val"], result["test"]
    print("end loading data....")
    args["model_params"]["mask_mlp_params"]["features_len"] = feature_len
    model = model_func(args["dataset_params"]["pre_len"],
                       args["dataset_params"]["suf_len"],
                       args["model_params"]["overall_params"],
                       args["model_params"]["multi_tcn_params"],
                       args["model_params"]["mask_mlp_params"],
                       args["model_params"]["lw_former_params"]
                       )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], betas=args["betas"], weight_decay=args["weight_decay"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
    model.to(device)
    criterion.to(device)
    num_epochs = args["num_epochs"]

    writer = None
    current_datetime = datetime.now()
    data_time = current_datetime.strftime("(%Y-%m-%d)-(%H-%M)-%S")

    if args["logs_dir"]:
        writer = SummaryWriter(args["logs_dir"] + "/" + data_time)

    for epoch in range(num_epochs):
        train_epoch_loss, train_epoch_acc = train_func(model, train_loader, criterion, optimizer, device, epoch,
                                                       num_epochs)
        result = test_func(model, test_loader, criterion, device)
        acc_ls.append(result)
        if args["logs_dir"]:
            writer.add_scalar('Loss/train_funcs', train_epoch_loss, epoch)
            writer.add_scalar('Accuracy/train_funcs', train_epoch_acc, epoch)

            writer.add_scalar('Loss/test', result[0], epoch)
            writer.add_scalar('Accuracy/test', result[1], epoch)
            writer.add_scalar('F1 Score', result[2], epoch)
            writer.add_scalar('Recall', result[3], epoch)
            writer.add_scalar('Precision', result[4], epoch)
    if args["logs_dir"]:
        writer.close()
    return acc_ls, data_time, args


if __name__ == '__main__':
    pass
