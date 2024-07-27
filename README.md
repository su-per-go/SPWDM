## Overview

SPWDM is a scalable model that performs well even in the presence of missing data. The model incorporates four types of inputs: URL prefix, URL suffix, handcrafted features, and handcrafted feature masks. It employs a dual-branch Temporal Convolutional Network (TCN) to extract sequential features from the URL prefix and suffix. By using masked attention, the model extends features across various dimensions, including URL statistical features, response features, HTML features, and dynamic features, thereby enhancing detection accuracy. The architecture of the detection model is illustrated in the following figure.

![1](https://github.com/su-per-go/SPDM/blob/master/1.png)

The [Crawling2024 dataset](https://github.com/su-per-go/SPDM/tree/master/dataset/untreated_dataset/crawling2024) can be found at `SPWDM/dataset/untreated_dataset/crawling2024` or downloaded [here](https://www.kaggle.com/datasets/haozhang1579/crawling-2024/). To enhance the dataset, we have included the raw captured data. Additionally, to facilitate dataset expansion, we provide the code for [data capture](https://github.com/su-per-go/crawling_url), [feature extraction, and dataset generation](https://github.com/su-per-go/feature_extra).The architecture for dataset construction is shown in the following figure.

![2](https://github.com/su-per-go/SPDM/blob/master/2.png)

# Project Structure

| Folder/File Name                                | Purpose                                                      |
| ----------------------------------------------- | ------------------------------------------------------------ |
| dataset/                                        | This folder contains datasets for comparative literature and Crawling2024 datasets, as well as some processing procedures for these datasets. |
| dataset/untreated_dataset/get_train_test_val.py | Responsible for dividing raw data into testing, training, and validation. |
| dataset/untreated_dataset/                      | Store unprocessed comparison dataset and Crawling2024 dataset. |
| dataset/PAGE_feature_extra.py                   | Responsible for extracting page features from the compared dataset. |
| dataset/URL_feature_extra.py                    | Responsible for extracting URL features from the compared dataset. |
| dataset/handel_data.py                          | Responsible for further extracting features from the initially divided test, training, and validation data, and subsequently generating the final test, training, and validation datasets. |
| model_file/                                     | Store the model proposed in this article.                    |
| output/                                         | Store the output results of various datasets, as well as the specific parameters during the experiment at that time. |
| train_funcs/                                    | Store information such as training functions and parameters  |
| load_data.py                                    | Responsible for loading datasets.                            |
| tests.py                                        | Do some testing to use.                                      |
| train.py                                        | Through this file, some parameters of the model and dataset can be adjusted to train the model and return the trained results. |

# Testing

You can complete the training, verification and testing of the model in train.py. You can also adjust the following parameters to adjust the model.

```python
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
```
