import json


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def _load_labeled_data(data_path, gold_path):
    data_json = load_json(data_path)
    labels_json = load_json(gold_path)

    labels_dict = {sample['id']: sample for sample in labels_json}
    for sample in data_json:
        sample['tag'] = labels_dict[sample['id']]['tag']

    return data_json


def _load_unlabeled_data(data_path):
    return load_json(data_path)


def _construct_samples(data_samples, vectors_path):
    vectors = load_json(vectors_path)
    vectors_dict = {pred['id']: pred for pred in vectors}

    for sample in data_samples:
        pred = vectors_dict[sample['id']]
        for key, value in pred.items():
            sample[key] = value

    return data_samples


def construct_labeled_samples(data_path, gold_path, vectors_path):
    data_samples = _load_labeled_data(data_path, gold_path)
    return _construct_samples(data_samples, vectors_path)


def construct_unlabeled_samples(data_path, vectors_path):
    data_samples = _load_unlabeled_data(data_path)
    return _construct_samples(data_samples, vectors_path)
