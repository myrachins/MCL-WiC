import argparse
from sklearn.metrics import accuracy_score
import pandas as pd
from utils.samples import load_json

parser = argparse.ArgumentParser(description='MCL-WiC GlossReader - test score')

parser.add_argument('--en-test-pred', type=str, default='data/preds/preds_en.data')
parser.add_argument('--ru-test-pred', type=str, default='data/preds/preds_ru.data')
parser.add_argument('--fr-test-pred', type=str, default='data/preds/preds_fr.data')
parser.add_argument('--ar-test-pred', type=str, default='data/preds/preds_ar.data')
parser.add_argument('--zh-test-pred', type=str, default='data/preds/preds_zh.data')

parser.add_argument('--en-test-gold', type=str, default='data/MCL-WiC/test/test-gold-data/test.en-en.gold')
parser.add_argument('--ru-test-gold', type=str, default='data/MCL-WiC/test/test-gold-data/test.ru-ru.gold')
parser.add_argument('--fr-test-gold', type=str, default='data/MCL-WiC/test/test-gold-data/test.fr-fr.gold')
parser.add_argument('--ar-test-gold', type=str, default='data/MCL-WiC/test/test-gold-data/test.ar-ar.gold')
parser.add_argument('--zh-test-gold', type=str, default='data/MCL-WiC/test/test-gold-data/test.zh-zh.gold')


def compute_accuracy(pred_filepath, gold_filepath):
    preds = load_json(pred_filepath)
    golds = load_json(gold_filepath)

    def transform_to_binary(json_samples):
        return [sample['tag'] == 'T' for sample in json_samples]

    y_preds = transform_to_binary(preds)
    y_golds = transform_to_binary(golds)

    return accuracy_score(y_golds, y_preds)


def calculate_test_scores(args):
    en_acc = compute_accuracy(args.en_test_pred, args.en_test_gold)
    ru_acc = compute_accuracy(args.ru_test_pred, args.ru_test_gold)
    fr_acc = compute_accuracy(args.fr_test_pred, args.fr_test_gold)
    ar_acc = compute_accuracy(args.ar_test_pred, args.ar_test_gold)
    zh_acc = compute_accuracy(args.zh_test_pred, args.zh_test_gold)

    results_table = pd.DataFrame({
        'en': en_acc,
        'ru': ru_acc,
        'fr': fr_acc,
        'ar': ar_acc,
        'zh': zh_acc
    }, index=[0])
    print(f'Test accuracy scores:', results_table.to_string(index=False), sep='\n')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    calculate_test_scores(args=args)
