import pandas as pd
from string import printable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import time


class LoadDataset:
    def __init__(self, path, is_read_url_features=True, is_read_page_features=True, is_read_res_features=True,
                 is_read_dyn_features=True, to_normalize=False, state_code=None):
        self.state_code = state_code  # 未设置state_code将读取特征中的res_state_code  值
        self.read_file = pd.read_csv(path)
        self.is_read_url_features = is_read_url_features
        self.is_read_page_features = is_read_page_features
        self.is_read_res_features = is_read_res_features
        self.is_read_dyn_features = is_read_dyn_features
        self.to_normalize = to_normalize
        self.feature_num = 0

    def read_url_features(self):
        return self.read_file.filter(regex='^url').drop(columns=["url_url", "url_url_pre", "url_url_suf"])

    def read_page_features(self):
        return self.read_file.filter(regex='^page')

    def read_res_features(self):
        return self.read_file.filter(regex='^res')

    def read_dyn_features(self):
        return self.read_file.filter(regex='^dyn')

    def read_url_pre(self):
        return self.read_file.filter(regex="^url_url_pre")

    def read_url_suf(self):
        return self.read_file.filter(regex="^url_url_suf")

    def read_url_url(self):
        return self.read_file["url_url"]

    # def read_max_len(self):
    #     return self.read_file.filter(regex="^pre_best_embedding_len").values[0][0], self.read_file.filter(
    #         regex="^suf_best_embedding_len").values[0][0]

    def read_features(self):
        csv_ls = []
        if self.is_read_url_features:
            csv_ls.append(self.read_url_features())

        if self.is_read_page_features:
            csv_ls.append(self.read_page_features())

        if self.is_read_res_features:
            csv_ls.append(self.read_res_features())

        if self.is_read_dyn_features:
            csv_ls.append(self.read_dyn_features())

        features = pd.concat(csv_ls, ignore_index=True, axis=1)
        if self.to_normalize:
            return normalize(features, 0, 100)
        else:
            return features

    def read_mask(self):
        mask_ls = []
        url_features_len = self.read_url_features().shape[1]
        page_features_len = self.read_page_features().shape[1]
        res_features_len = self.read_res_features().shape[1]
        dyn_features_len = self.read_dyn_features().shape[1]
        if "res_state_code" in self.read_res_features().columns:
            the_index = self.read_res_features().columns.get_loc("res_state_code")
        else:
            the_index = -1
        for row in self.read_file.itertuples():
            if self.state_code:
                res_state_code = self.state_code
            else:
                res_state_code = row.res_state_code
            url_mask = []
            if self.is_read_url_features:
                url_mask.extend([0] * url_features_len)
            if self.is_read_page_features:
                if 300 > res_state_code >= 200:
                    url_mask.extend([0] * page_features_len)
                else:
                    url_mask.extend([1] * page_features_len)
            if self.is_read_res_features:
                if 300 > res_state_code >= 200:
                    url_mask.extend([0] * res_features_len)
                elif res_state_code == 0:
                    url_mask.extend([1] * res_features_len)
                else:
                    the_mask = [1] * res_features_len
                    the_mask[the_index] = 0
                    url_mask.extend(the_mask)
            if self.is_read_dyn_features:
                if 300 > res_state_code >= 200:
                    url_mask.extend([0] * dyn_features_len)
                else:
                    url_mask.extend([1] * dyn_features_len)
            mask_ls.append(url_mask)

        return mask_ls

    def read_label(self):
        return self.read_file.filter(regex="^label")

    def dispatch(self):
        return self.read_url_url(), self.read_url_pre(), self.read_url_suf(), self.read_features(), self.read_mask(), self.read_label(),


def normalize(data, begin, end):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = pd.DataFrame()
    for col in data.columns:
        if min_vals[col] == max_vals[col]:  # 如果最小值和最大值相等，则直接将归一化后的值设置为零
            normalized_data[col] = 0
        else:
            normalized_data[col] = (data[col] - min_vals[col]) / (max_vals[col] - min_vals[col]) * (end - begin) + begin

    # 处理NaN和inf值
    normalized_data = normalized_data.fillna(0)  # 将NaN值填充为0
    normalized_data = normalized_data.round().astype(int)  # 将数据四舍五入并转换为整数
    return normalized_data


class CreateDataset:
    def __init__(self, url_data, pre_data, suf_data, features_data, mask_data, label_data, url_len, pre_len, suf_len,
                 fill_style):
        self.url_data = url_data
        self.pre_data = pre_data
        self.suf_data = suf_data
        self.features_data = features_data
        self.mask_data = mask_data
        self.label_data = label_data
        self.url_len = url_len
        self.pre_len = pre_len
        self.suf_len = suf_len
        self.fill_style = fill_style

    @staticmethod
    def seq_operate(token, end, offset=0, fill_sty="pre"):  # 获取训练长度
        seq_ls = []
        for seq in token:
            if fill_sty == "pre":
                seq_ls.append(torch.LongTensor([0] * min(end, (end - len(seq) + offset)) + seq[offset:end + offset]))
            elif fill_sty == "suf":
                seq_ls.append(torch.LongTensor(seq[offset:end + offset] + [0] * min(end, (end - len(seq) + offset))))
        return pad_sequence(seq_ls, batch_first=True)

    @staticmethod
    def encod_urls(urls_ls):  # 编码 URL
        return [[printable.index(x) + 1 for x in str(url) if x in printable] for url in urls_ls]

    def create(self):
        pre_data = self.seq_operate(self.encod_urls(self.pre_data.values.ravel().tolist()), self.pre_len,
                                    fill_sty=self.fill_style[0])  # 前缀
        suf_data = self.seq_operate(self.encod_urls(self.suf_data.values.ravel().tolist()), self.suf_len,
                                    fill_sty=self.fill_style[1])  # 后缀
        url_data = self.seq_operate(self.encod_urls(self.url_data.values.ravel().tolist()), self.url_len,
                                    fill_sty="suf")

        class CustomDataset(Dataset):
            def __init__(self, url_url_data, url_pre_data, url_suf_data, features_data, mask_data, label_data):
                self.url_url_data = url_url_data
                self.url_pre_data = url_pre_data
                self.url_suf_data = url_suf_data

                self.features_data = torch.tensor(features_data.values, dtype=torch.float)
                self.mask_data = torch.tensor(mask_data, dtype=torch.float)
                self.label_data = torch.tensor(label_data.values, dtype=torch.int)

            def __len__(self):
                return len(self.label_data)

            def __getitem__(self, idx):
                sample = {
                    "url_url": self.url_url_data[idx],
                    "url_pre": self.url_pre_data[idx],
                    "url_suf": self.url_suf_data[idx],
                    "features": self.features_data[idx],
                    "mask": self.mask_data[idx],
                    "label": self.label_data[idx]
                }
                return sample

        feature_len = self.features_data.shape[1]
        return CustomDataset(url_data, pre_data, suf_data, self.features_data, self.mask_data,
                             self.label_data), feature_len


def get_dataset(**kwargs):
    if kwargs["dataset_path"] is None:
        raise ValueError("dataset_path is required")
    if kwargs["batch_size"] is None:
        raise ValueError("batch_size is required")

    result_ls = {}
    if kwargs["train_type"] == "train_val_test":
        dataset_ls = ["train", "val", "test"]
    elif kwargs["train_type"] == "train_test":
        dataset_ls = ["train", "test"]
        result_ls["val"] = []
    else:
        dataset_ls = []
    feature_len = 0
    for data in dataset_ls:
        csv_path = kwargs["dataset_path"] + "/" + data + ".csv"
        custom_dataset, feature_len = CreateDataset(*LoadDataset(csv_path, **kwargs["is_read"]).dispatch(),
                                                    url_len=kwargs["url_len"],
                                                    pre_len=kwargs["pre_len"],
                                                    suf_len=kwargs["suf_len"],
                                                    fill_style=kwargs["fill_style"]).create()
        result_ls[data] = (DataLoader(custom_dataset, batch_size=kwargs["batch_size"], shuffle=kwargs["shuffle"]))
    return result_ls, feature_len


if __name__ == "__main__":
    dataset_kwargs = {
        "dataset_path": "dataset/crawling2024/",
        "batch_size": 128,
        "is_read": {"is_read_url_features": True, "is_read_page_features": False, "is_read_res_features": False,
                    "is_read_dyn_features": False, "to_normalize": True, "state_code": None},
        "url_len": 150, "pre_len": 50, "suf_len": 50, "shuffle": False, "train_type": "train_val_test",
        "fill_style": ("pre", "suf")
    }
    result, feature_length = get_dataset(**dataset_kwargs)
    for i in result["train"]:
        print(i["mask"].shape)
        for j in i["mask"][:-10]:
            print(j)
            pass
        break
    print(result)
