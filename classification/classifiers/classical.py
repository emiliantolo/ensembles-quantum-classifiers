import numpy as np
from classification.helpers.distances import cosine_sim, euclidean_dist
from classification.base import BaseClassifier


class CosineClassifier(BaseClassifier):

    def __init__(self, data):
        BaseClassifier.__init__(self, data)

    def classify(self, x):
        sum = 0
        for d in self.data:
            sum += d['y'] * cosine_sim(x, d['x'])
        conf = np.abs(sum / len(self.data))
        return np.sign(sum), conf


class DistanceClassifier(BaseClassifier):

    def __init__(self, data, norm=True):
        BaseClassifier.__init__(self, data)
        self.norm = norm

    def classify(self, x):
        sum = 0
        for d in self.data:
            sum += d['y'] * (1 - 1 / 4 * pow(euclidean_dist(x,
                             d['x'], norm=self.norm), 2))
        conf = np.abs(sum / len(self.data))
        return np.sign(sum), conf


class KNNCosineSquaredClassifier(BaseClassifier):

    def __init__(self, data, k=1):
        BaseClassifier.__init__(self, data)
        self.k = k

    def classify(self, x):
        ratings = []
        for d in self.data:
            r = pow(cosine_sim(x, d['x']), 2)
            ratings.append((r, d['y']))
        top_k = sorted(ratings, key=lambda t: t[0], reverse=True)[:self.k]
        top_classes = [t[1] for t in top_k]
        val, cnt = np.unique(top_classes, return_counts=True)
        counts = dict(zip(val, cnt))
        top_class = val[0]
        top_count = cnt[0]
        for i in reversed(range(self.k)):
            c = top_classes[i]
            if counts[c] >= top_count:
                top_count = counts[c]
                top_class = c
        conf = top_count / self.k
        return top_class, conf
