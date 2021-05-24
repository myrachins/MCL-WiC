from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils.samples import construct_labeled_samples


class ThresholdSelector(ABC):
    @abstractmethod
    def get_threshold(self, lang):
        pass

    @staticmethod
    def calculate_best_threshold(similarity_measurer, samples, thresholds):
        y_true = [sample['tag'] == 'T' for sample in samples]
        scores = []

        for threshold in tqdm(thresholds):
            y_pred = similarity_measurer.predict_samples(samples, threshold)
            scores.append(accuracy_score(y_true, y_pred))

        return thresholds[np.argmax(scores)]


class SingleThresholdSelector(ThresholdSelector):
    def __init__(self, similarity_measurer, data_path, gold_path, vectors_path, grid_start, grid_stop, grid_num):
        samples = construct_labeled_samples(data_path, gold_path, vectors_path)
        thresholds = np.linspace(grid_start, grid_stop, grid_num)
        self.threshold = self.calculate_best_threshold(similarity_measurer, samples, thresholds)

    def get_threshold(self, lang):
        return self.threshold


class XXThresholdSelector(ThresholdSelector):
    def __init__(self, lang_to_threshold_selector):
        self.lang_to_threshold_selector = lang_to_threshold_selector

    def get_threshold(self, lang):
        return self.lang_to_threshold_selector[lang].get_threshold(lang)


