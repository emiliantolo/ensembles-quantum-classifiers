import numpy as np
from classification.base import BaseClassifier
from classification.helpers.factories import gen_classifier
from classification.helpers.data import get_samples


class BoostingClassifier(BaseClassifier):

    def __init__(self, data, classifier_name='cosine', n_samples=10, N=100, balanced=False, **cl_args):
        BaseClassifier.__init__(self, data)
        self.classifier_name = classifier_name
        self.n_samples = n_samples
        self.N = N
        self.balanced = balanced
        self.classifiers = []
        self.alpha = []
        self.cl_args = cl_args

    def build(self):
        eps = 1e-10
        weights = np.array([1 / len(self.data)] * len(self.data))
        for i in range(self.N):
            data = get_samples(
                self.data, self.n_samples, replace=True, p=weights, balanced=self.balanced)
            classifier = gen_classifier(
                self.classifier_name, data, **self.cl_args)
            predict = np.array([classifier.classify(d['x'])[0]
                               for d in self.data])
            label = np.array([d['y'] for d in self.data])
            delta = np.array([(predict[i] != label[i])
                              for i in range(len(label))])
            epsilon = np.sum(weights * delta) / np.sum(weights) + eps
            alpha = np.log((1 - epsilon) / epsilon)
            weights = weights * (np.exp(alpha * delta))
            weights = weights / np.sum(weights)
            self.alpha.append(alpha)
            self.classifiers.append(classifier)

    def classify(self, x):
        sum = 0
        for i in range(self.N):
            sum += self.alpha[i] * self.classifiers[i].classify(x)[0]
        return np.sign(sum), np.abs(sum) / np.sum(np.abs(self.alpha))
