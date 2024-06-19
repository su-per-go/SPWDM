import pandas as pd
from sklearn.model_selection import train_test_split

from dataset.untreated_dataset.PSU.get_same_dataset import get_same_dataset

SEED = 42


def get_train_test_value(read_path, save_path, is_save=False, same_data_fuc=None):
    if same_data_fuc:
        df = get_same_dataset(read_path, SEED)
    else:
        df = pd.read_csv(read_path)
    labels = df["label"]
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    urls = df.drop(columns=['label'])
    X_train_val, X_test, y_train_val, y_test = train_test_split(urls, labels, test_size=0.1,
                                                                random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=SEED)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    # 保存训练集和测试集为 CSV 文件
    if is_save:
        train_df.to_csv(save_path + '/train.csv', index=False)
        test_df.to_csv(save_path + '/test.csv', index=False)
        val_df.to_csv(save_path + '/val.csv', index=False)
    else:
        return train_df, test_df, val_df


if __name__ == '__main__':
    get_train_test_value("PSU/phishing_site_urls.csv", "PSU", is_save=True, same_data_fuc=get_same_dataset
                         )
