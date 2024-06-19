import pandas as pd


def get_same_dataset(read_path, seed):
    result_csv = pd.read_csv(read_path, )
    result_csv.rename(columns={"URL": "url", "Label": "label"}, inplace=True)
    result_csv['label'] = result_csv['label'].replace({'bad': 1, 'good': 0})
    bad_rows = result_csv[result_csv['label'] == 0]
    good_rows = result_csv[result_csv['label'] == 1]
    min_samples = min(len(bad_rows), len(good_rows))
    bad_rows = bad_rows.sample(n=min_samples, random_state=seed)
    good_rows = good_rows.sample(n=min_samples, random_state=seed)
    result_csv = pd.concat([bad_rows, good_rows])
    return result_csv
