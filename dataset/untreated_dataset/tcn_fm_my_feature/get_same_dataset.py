import pandas as pd


def get_same_dataset(read_path, seed):
    result_csv = pd.read_csv(read_path, )
    result_csv.rename(columns={"url_url": "url", }, inplace=True)
    urls = result_csv["url"]
    labels = result_csv["label"]
    result_csv = pd.concat([urls, labels], axis=1)
    return result_csv
