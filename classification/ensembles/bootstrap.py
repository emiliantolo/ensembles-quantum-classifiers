import numpy as np
from classification.base import BaseClassifier
from classification.helpers.factories import gen_classifier
from classification.helpers.data import get_samples


class BootstrapClassifier(BaseClassifier):

    def __init__(self, data, classifier_name='cosine', policy_name='majority', n_samples=10, N=100, balanced=False, **cl_args):
        BaseClassifier.__init__(self, data)
        self.classifier_name = classifier_name
        self.policy_name = policy_name
        self.n_samples = n_samples
        self.N = N
        self.balanced = balanced
        self.classifiers = []
        self.cl_args = cl_args

    def build(self):
        for i in range(self.N):
            data = get_samples(self.data, self.n_samples, replace=True, balanced=self.balanced)
            classifier = gen_classifier(
                self.classifier_name, data, **self.cl_args)
            self.classifiers.append(classifier)

    def classify_all(self, x):
        res = []
        conf = []
        for i in range(self.N):
            r, c = self.classifiers[i].classify(x)
            res.append(r)
            conf.append(c)
        return np.array(res), np.array(conf)

    def classify(self, x):
        res, conf = self.classify_all(x)
        if self.policy_name == 'majority':
            val, cnt = np.unique(res, return_counts=True)
            return val[np.argmax(cnt)], np.max(cnt) / self.N
        elif self.policy_name == 'weight_lin':
            val = np.unique(res)
            cnt = {}
            for v in val:
                cnt[v] = 0
            for i in range(self.N):
                cnt[res[i]] += conf[i]
            c = max(cnt.keys(), key=lambda k: cnt[k])
            w = cnt[c]
            return c, w / self.N
        elif self.policy_name == 'weight_soft':
            e_x = np.exp(conf - np.max(conf))
            conf = e_x / e_x.sum(axis=0)
            val = np.unique(res)
            cnt = {}
            for v in val:
                cnt[v] = 0
            for i in range(self.N):
                cnt[res[i]] += conf[i]
            c = max(cnt.keys(), key=lambda k: cnt[k])
            w = cnt[c]
            return c, w / self.N
        else:
            raise ValueError('policy_name not recognized')

