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


if __name__ == '__main__':
    SEED = 42
    train_csv = read_dataset("train.csv")
    test_csv = read_dataset("test.csv")
    labels = test_csv["label"]
    urls = test_csv.drop(columns=['label'])
    X_test, X_val, y_test, y_val = train_test_split(urls, labels, test_size=0.5,
                                                    random_state=SEED)
    test_df = pd.concat([X_test, y_test], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df.to_csv("test.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    train_csv.to_csv("train.csv", index=False)
