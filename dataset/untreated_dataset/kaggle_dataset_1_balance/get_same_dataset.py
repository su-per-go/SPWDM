import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder


def read_dataset(file_path, max_sequence_length=None, convert_to_array=True, label_phish=1, label_legit=0,
                 split_char='\t'):
    with open(file_path, encoding="utf8", errors='ignore') as fp:
        line = fp.readline()
        le = LabelEncoder()
        samples = []
        labels = []
        while line:
            if line.__contains__('\n'):
                line = line.replace('\n', '')
                line = line.replace(' ', '')
            parts = line.split(split_char)
            if len(parts) == 1:
                parts = parts[0].split(",")
            label = int(parts[0])
            if label == 2 or label == -1:
                label = label_legit
            else:
                label = label_phish
            sample = parts[1]

            if len(sample) < 8:
                line = fp.readline()
                continue
            if max_sequence_length is not None:
                sample = sample[:max_sequence_length]
            samples.append(sample)
            labels.append(label)

            line = fp.readline()
        df = pd.DataFrame()
        df["url"] = samples
        df["label"] = labels
        return df


def to_balance(csv_file, seed):
    bad_rows = csv_file[csv_file['label'] == 0]
    good_rows = csv_file[csv_file['label'] == 1]
    min_samples = min(len(bad_rows), len(good_rows))
    bad_rows = bad_rows.sample(n=min_samples, random_state=seed)
    good_rows = good_rows.sample(n=min_samples, random_state=seed)
    result_csv = pd.concat([bad_rows, good_rows])
    return result_csv


if __name__ == '__main__':
    SEED = 42
    train_csv = read_dataset("train_from_gram.csv")
    test_csv = read_dataset("test_from_gram.csv")
    train_csv = to_balance(train_csv, seed=SEED)
    test_csv = to_balance(test_csv, seed=SEED)

    labels = test_csv["label"]
    urls = test_csv.drop(columns=['label'])

    X_test, X_val, y_test, y_val = train_test_split(urls, labels, test_size=0.5,
                                                    random_state=SEED)
    test_df = pd.concat([X_test, y_test], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df.to_csv("test.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    train_csv.to_csv("train.csv", index=False)
