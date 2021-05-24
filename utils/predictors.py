from abc import ABC, abstractmethod
import numpy as np


class SimilarityPredictor:
    def __init__(self, threshold_selector, similarity_measurer):
        self.threshold_selector = threshold_selector
        self.similarity_measurer = similarity_measurer

    def predict(self, out_vector_1, out_vector_2, lang):
        threshold = self.threshold_selector.get_threshold(lang)
        return self.similarity_measurer.predict(out_vector_1, out_vector_2, threshold)

    def predict_samples(self, samples, lang):
        return [self.predict(sample['context_output1'], sample['context_output2'], lang) for sample in samples]


class SimilarityMeasurer(ABC):
    @abstractmethod
    def similarity(self, out_vector_1, out_vector_2):
        pass

    @abstractmethod
    def predict(self, out_vector_1, out_vector_2, threshold):
        pass

    def predict_samples(self, samples, threshold):
        return [self.predict(sample['context_output1'], sample['context_output2'], threshold) for sample in samples]


class VectorsDistSimilarity(SimilarityMeasurer):
    def __init__(self, normalize=True, norm_ord=2):
        self.normalize = normalize
        self.norm_ord = norm_ord

    def similarity(self, out_vector_1, out_vector_2):
        out_vector_1 = np.array(out_vector_1)
        out_vector_2 = np.array(out_vector_2)

        if self.normalize:
            out_vector_1 /= np.linalg.norm(out_vector_1, ord=self.norm_ord)
            out_vector_2 /= np.linalg.norm(out_vector_2, ord=self.norm_ord)

        return np.linalg.norm(out_vector_1 - out_vector_2, ord=self.norm_ord)

    def predict(self, out_vector_1, out_vector_2, threshold):
        return self.similarity(out_vector_1, out_vector_2) < threshold
