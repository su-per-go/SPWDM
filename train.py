import json
import os
import time

from model_file.model import DetectionModel
from train_funcs.train_param import train_1_param

bast_path = os.getcwd()


def sava_params(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def run(the_dataset):
    args_dict = {
        "model": "model",
        "num_epochs": 150,
        "lr": 1e-3,
        "betas": (0.90, 0.99),
        "weight_decay": 1e-4,
        "logs_dir": bast_path + "/output/" + the_dataset,
        "dataset_params": {
            "dataset_path": bast_path + "/dataset/" + the_dataset,
            "batch_size": 1024,
            "is_read": {"is_read_url_features": True, "is_read_page_features": True, "is_read_res_features": True,
                        "is_read_dyn_features": True, "to_normalize": True, "state_code": None},
            "url_len": 150, "pre_len": 50, "suf_len": 50, "shuffle": True, "train_type": "train_val_test",
            "fill_style": ("pre", "suf")
        },
        "model_params": {
            "the_max_pool_output_size": 16,
            "tcn_params": {
                "pre_params": {
                    "input_size": 16,
                    "output_size": 128,
                    "num_channels": [64, 128],
                    "kernel_size": [3, 5],
                    "dropout": 0.1
                },
                "suf_params": {
                    "input_size": 16,
                    "output_size": 128,
                    "num_channels": [64, 128],
                    "kernel_size": [3, 5],
                    "dropout": 0.1
                }
            },
            "mask_atte_params": {
                "hidden_dim": 128,
                "num_heads": 16
            },
            "sc_attention_params": {
                "in_size": 32,
                "local_size": 8,
                "local_weight": 0.5,
                "global_weight": 0.5
            },
            "feature_fusion_params": {
                "input_dim": 128,
                "embed_dim": 64,
                "pool_dim": 2
            }
        },

    }
    ls = []
    for i in range(5):
        acc_ls, data_time, args_dict = train_1_param(DetectionModel, args_dict)
        acc = []
        for info in acc_ls:
            acc.append(info[1])
        print(i, max(acc), "最大精度")
        ls.append(max(acc))
        sava_params(args_dict["logs_dir"] + "/" + data_time + "/" + str(max(acc) * 100000 // 1) + ".json",
                    {"args_dict": args_dict})
        time.sleep(6)
    print(ls)


if __name__ == '__main__':
    dataset_ls = [
        "crawling2024",
    ]
    for i in dataset_ls:
        run(i)
