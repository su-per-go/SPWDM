from urllib.parse import urlparse

import pandas as pd


def get_url_and_feature_csv(url_path, feature_path, label):
    with open(url_path, 'r', encoding="utf8") as f:
        url_ls = f.readlines()
        if url_path == "neg_url/domain_exist_at_nec.txt":
            url_ls = url_ls[:373946]
        url_ls = [[line.strip()] + split_url(line.strip()) for line in url_ls]
    urls = pd.DataFrame(url_ls, columns=['url_url', "url_url_pre", "url_url_suf"])

    features = pd.read_csv(feature_path)
    if url_path == "neg_url/domain_exist_at_nec.txt":
        features = features[:373946]
    new_column_names = ["url_" + name for name in features.columns]
    features.columns = new_column_names

    label_column = pd.DataFrame({'label': [1] * len(urls)})
    data = pd.concat([urls, features, label_column], axis=1)
    return data


def split_url(url):
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc
    # 提取路径
    path = parsed_url.path
    if len(domain_name) == 0:
        path = path.lstrip("/")
        sp_url = path.split("/", 1)
        prefix = sp_url[0]
        suffix = sp_url[1] if len(sp_url) > 1 else ""
    else:
        prefix = parsed_url.scheme + "://" + parsed_url.netloc
        suffix = parsed_url.path + parsed_url.params + parsed_url.query + parsed_url.fragment
    return [prefix, suffix]

if __name__ == '__main__':
    dataset_ls = [
        ['neg_url/domain_exist_at_nec.txt', "neg_url_widefeature/domain_exist_at_net_wideFeature.csv", 1],

        ['neg_url/domain_is_ip.txt', "neg_url_widefeature/domain_is_ip_wideFeature.csv", 1],

        ['neg_url/real_domain_exist_at_nec_and_others.txt',
         "neg_url_widefeature/real_domain_exist_at_nec_and_others_wideFeature.csv", 1],

        ['neg_url/real_domain_exist_at_others.txt',
         "neg_url_widefeature/real_domain_exist_at_others_wideFeature.csv", 1],

        ['neg_url/url_no_obvious_domain.txt',
         "neg_url_widefeature/url_no_obvious_domain_wideFeature.csv", 1],

        ['pos_url/pos_url.txt',
         "pos_url_widefeature/pos_url_wideFeature.csv", 0],
    ]
    data_result = []
    for i in dataset_ls:
        data_result.append(get_url_and_feature_csv(i[0], i[1], i[2]))
    dataset = pd.concat(data_result, axis=0)
    dataset.to_csv('dataset.csv',index=False)
