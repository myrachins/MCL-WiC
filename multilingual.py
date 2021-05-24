import argparse
import json
import os
from utils.samples import construct_unlabeled_samples
from utils.thresholds import SingleThresholdSelector, XXThresholdSelector
from utils import predictors

parser = argparse.ArgumentParser(description='MCL-WiC GlossReader - multilingual')
parser.add_argument('--similarity', type=str, default='l1+norm',
                    choices=['l1', 'l1+norm', 'l2', 'l2+norm'])
parser.add_argument('--threshold', type=str, default='en-dev',
                    choices=['en-dev', 'xx-dev', 'semcor'])

parser.add_argument('--en-dev-data', type=str, default='data/MCL-WiC/dev/multilingual/dev.en-en.data')
parser.add_argument('--en-dev-gold', type=str, default='data/MCL-WiC/dev/multilingual/dev.en-en.gold')
parser.add_argument('--ru-dev-data', type=str, default='data/MCL-WiC/dev/multilingual/dev.ru-ru.data')
parser.add_argument('--ru-dev-gold', type=str, default='data/MCL-WiC/dev/multilingual/dev.ru-ru.gold')
parser.add_argument('--fr-dev-data', type=str, default='data/MCL-WiC/dev/multilingual/dev.fr-fr.data')
parser.add_argument('--fr-dev-gold', type=str, default='data/MCL-WiC/dev/multilingual/dev.fr-fr.gold')
parser.add_argument('--ar-dev-data', type=str, default='data/MCL-WiC/dev/multilingual/dev.ar-ar.data')
parser.add_argument('--ar-dev-gold', type=str, default='data/MCL-WiC/dev/multilingual/dev.ar-ar.gold')
parser.add_argument('--zh-dev-data', type=str, default='data/MCL-WiC/dev/multilingual/dev.zh-zh.data')
parser.add_argument('--zh-dev-gold', type=str, default='data/MCL-WiC/dev/multilingual/dev.zh-zh.gold')

parser.add_argument('--en-test-data', type=str, default='data/MCL-WiC/test/multilingual/test.en-en.data')
parser.add_argument('--ru-test-data', type=str, default='data/MCL-WiC/test/multilingual/test.ru-ru.data')
parser.add_argument('--fr-test-data', type=str, default='data/MCL-WiC/test/multilingual/test.fr-fr.data')
parser.add_argument('--ar-test-data', type=str, default='data/MCL-WiC/test/multilingual/test.ar-ar.data')
parser.add_argument('--zh-test-data', type=str, default='data/MCL-WiC/test/multilingual/test.zh-zh.data')

parser.add_argument('--en-dev-vectors', type=str, default='data/vectors-glm-large/dev/vectors_dev.en-en.data')
parser.add_argument('--ru-dev-vectors', type=str, default='data/vectors-glm-large/dev/vectors_dev.ru-ru.data')
parser.add_argument('--fr-dev-vectors', type=str, default='data/vectors-glm-large/dev/vectors_dev.fr-fr.data')
parser.add_argument('--ar-dev-vectors', type=str, default='data/vectors-glm-large/dev/vectors_dev.ar-ar.data')
parser.add_argument('--zh-dev-vectors', type=str, default='data/vectors-glm-large/dev/vectors_dev.zh-zh.data')

parser.add_argument('--en-test-vectors', type=str, default='data/vectors-glm-large/test/vectors_test.en-en.data')
parser.add_argument('--ru-test-vectors', type=str, default='data/vectors-glm-large/test/vectors_test.ru-ru.data')
parser.add_argument('--fr-test-vectors', type=str, default='data/vectors-glm-large/test/vectors_test.fr-fr.data')
parser.add_argument('--ar-test-vectors', type=str, default='data/vectors-glm-large/test/vectors_test.ar-ar.data')
parser.add_argument('--zh-test-vectors', type=str, default='data/vectors-glm-large/test/vectors_test.zh-zh.data')

parser.add_argument('--semcor-data', type=str, default='data/SemCor/semcor_cut.data')
parser.add_argument('--semcor-gold', type=str, default='data/SemCor/semcor_cut.gold')
parser.add_argument('--semcor-vectors', type=str, default='data/vectors-glm-large/semcor/vectors_semcor.data')

parser.add_argument('--threshold-grid-start', type=float, default=0)
parser.add_argument('--threshold-grid-stop', type=float, default=1)
parser.add_argument('--threshold-grid-num', type=int, default=100)

parser.add_argument('--output-dir', type=str, default='data/preds')


def get_similarity_measurer(similarity):
    if similarity == 'l1+norm':
        return predictors.VectorsDistSimilarity(normalize=True, norm_ord=1)
    if similarity == 'l2+norm':
        return predictors.VectorsDistSimilarity(normalize=True, norm_ord=2)
    if similarity == 'l1':
        return predictors.VectorsDistSimilarity(normalize=False, norm_ord=1)
    if similarity == 'l2':
        return predictors.VectorsDistSimilarity(normalize=False, norm_ord=2)

    raise RuntimeError(f'Not supported similarity type: "{similarity}"')


def get_threshold_selector(similarity_measurer, args):
    threshold_grid_params = (args.threshold_grid_start, args.threshold_grid_stop, args.threshold_grid_num)
    if args.threshold == 'en-dev':
        return SingleThresholdSelector(
            similarity_measurer, args.en_dev_data, args.en_dev_gold, args.en_dev_vectors, *threshold_grid_params
        )
    if args.threshold == 'semcor':
        return SingleThresholdSelector(
            similarity_measurer, args.semcor_data, args.semcor_gold, args.semcor_vectors, *threshold_grid_params
        )
    if args.threshold == 'xx-dev':
        lang_to_threshold_selector = {
            'en': SingleThresholdSelector(
                similarity_measurer, args.en_dev_data, args.en_dev_gold, args.en_dev_vectors, *threshold_grid_params
            ),
            'ru': SingleThresholdSelector(
                similarity_measurer, args.ru_dev_data, args.ru_dev_gold, args.ru_dev_vectors, *threshold_grid_params
            ),
            'fr': SingleThresholdSelector(
                similarity_measurer, args.fr_dev_data, args.fr_dev_gold, args.fr_dev_vectors, *threshold_grid_params
            ),
            'ar': SingleThresholdSelector(
                similarity_measurer, args.ar_dev_data, args.ar_dev_gold, args.ar_dev_vectors, *threshold_grid_params
            ),
            'zh': SingleThresholdSelector(
                similarity_measurer, args.zh_dev_data, args.zh_dev_gold, args.zh_dev_vectors, *threshold_grid_params
            )
        }
        return XXThresholdSelector(lang_to_threshold_selector)

    raise RuntimeError(f'Not supported threshold type: "{args.threshold}"')


def save_predictions(samples, preds, lang, output_dir):
    predictions_filepath = os.path.join(output_dir, f'preds_{lang}.data')
    json_predictions = []

    for sample, pred in zip(samples, preds):
        json_predictions.append({
            'id': sample['id'],
            'tag': 'T' if pred else 'F'
        })

    with open(predictions_filepath, 'w') as f:
        json.dump(json_predictions, f, indent=4)


def generate_predictions(args):
    print('Loading test samples for all languages...')
    en_test_samples = construct_unlabeled_samples(args.en_test_data, args.en_test_vectors)
    ru_test_samples = construct_unlabeled_samples(args.ru_test_data, args.ru_test_vectors)
    fr_test_samples = construct_unlabeled_samples(args.fr_test_data, args.fr_test_vectors)
    ar_test_samples = construct_unlabeled_samples(args.ar_test_data, args.ar_test_vectors)
    zh_test_samples = construct_unlabeled_samples(args.zh_test_data, args.zh_test_vectors)

    print('Selecting best threshold...')
    similarity_measurer = get_similarity_measurer(args.similarity)
    threshold_selector = get_threshold_selector(similarity_measurer, args)
    predictor = predictors.SimilarityPredictor(threshold_selector, similarity_measurer)

    print('Predicting labels for test samples...')
    en_test_preds = predictor.predict_samples(en_test_samples, lang='en')
    ru_test_preds = predictor.predict_samples(ru_test_samples, lang='ru')
    fr_test_preds = predictor.predict_samples(fr_test_samples, lang='fr')
    ar_test_preds = predictor.predict_samples(ar_test_samples, lang='ar')
    zh_test_preds = predictor.predict_samples(zh_test_samples, lang='zh')

    print('Saving predictions...')
    save_predictions(en_test_samples, en_test_preds, lang='en', output_dir=args.output_dir)
    save_predictions(ru_test_samples, ru_test_preds, lang='ru', output_dir=args.output_dir)
    save_predictions(fr_test_samples, fr_test_preds, lang='fr', output_dir=args.output_dir)
    save_predictions(ar_test_samples, ar_test_preds, lang='ar', output_dir=args.output_dir)
    save_predictions(zh_test_samples, zh_test_preds, lang='zh', output_dir=args.output_dir)

    print('Predictions were saved!')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    generate_predictions(args=args)
