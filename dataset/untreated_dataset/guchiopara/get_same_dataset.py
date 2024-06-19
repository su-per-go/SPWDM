import pandas as pd


def get_same_dataset(read_path, seed):
    result_csv = pd.read_excel(read_path, )

    result_csv.rename(columns={"Category": "label", "Data": "url"}, inplace=True)
    html_csv = pd.read_excel("guchiopara/html.xlsx", )["Data"]
    result_csv['label'] = result_csv['label'].replace({'spam': 1, 'ham': 0})
    result_csv = pd.concat([result_csv, html_csv], axis=1)
    return result_csv
