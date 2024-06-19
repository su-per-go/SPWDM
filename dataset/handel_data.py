import os
import shutil

import pandas as pd

from dataset.PAGE_feature_extra import PageFeatureExtra
from dataset.URL_feature_extra import URLFeatureExtra


def get_subdirectories(folder_path):
    subdirectories = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item)
    return subdirectories


def handel_dataset(folder_path, dataset_names, url_feature=False, page_feature=False):
    subdirectories = get_subdirectories(folder_path)
    for subdirectory in subdirectories:
        if subdirectory in dataset_names:
            if os.path.exists(subdirectory):
                shutil.rmtree(subdirectory)
            os.mkdir(subdirectory)
            for file in ["train.csv", "test.csv", "val.csv"]:
                df = pd.read_csv(folder_path + "/" + subdirectory + "/" + file)
                if url_feature or page_feature:
                    url_ls = []
                    for index, row in enumerate(df.itertuples()):
                        feat_dict = {}
                        if url_feature:
                            feat_dict.update(URLFeatureExtra(row.url, row.label).handle())
                        if page_feature:
                            feat_dict.update(PageFeatureExtra(row.Data, row.url, is_path=False).handle())
                        url_ls.append(feat_dict)
                    df = pd.DataFrame(url_ls)
                df.to_csv(subdirectory + "/" + file, index=False)


if __name__ == '__main__':
    names = ["ebb2017", ]
    handel_dataset("untreated_dataset", names,url_feature=True)
