from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42


def process(file_path, max_sequence_length=None, convert_to_array=True, label_phish=1, label_legit=0,
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
            elif label == 1:
                label = label_phish
            sample = parts[1]

            if len(sample) < 8:
                line = fp.readline()
                continue

            if max_sequence_length is not None:
                sample = sample[:max_sequence_length]

            # sample = sample[1:]   # Remove first char from url because it is "
            # sample = sample[:-1]  # Remove last  char from url because it is "
            # sample = sample.lower()
            samples.append(sample)
            labels.append(label)

            line = fp.readline()
        return samples, labels


def split_test_val():
    urls, labels = process("test.csv")
    X_val, X_test, y_val, y_test = train_test_split(urls, labels, test_size=0.5, random_state=SEED)
    test_df = pd.DataFrame()
    test_df['url'] = X_test
    test_df['label'] = y_test
    test_df.to_csv('test.csv', index=False)

    val_df = pd.DataFrame()
    val_df['url'] = X_val
    val_df['label'] = y_val

    val_df.to_csv('val.csv', index=False)


def get_train():
    urls, labels = process("train.csv")
    train_df = pd.DataFrame()
    train_df['url'] = urls
    train_df['label'] = labels
    train_df.to_csv('train.csv', index=False)


if __name__ == "__main__":
    get_train()
    split_test_val()
