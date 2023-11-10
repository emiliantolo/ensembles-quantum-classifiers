import numpy as np
from classification.base import BaseClassifier
from classification.helpers.factories import gen_classifier


class StackingClassifier(BaseClassifier):

    def __init__(self, data, classifiers=[{'name': 'cosine', 'args': {}}], meta_classifier={'name': 'cosine', 'args': {}}, folds=5):
        BaseClassifier.__init__(self, data)
        self.classifier_names = classifiers
        self.meta_classifier_name = meta_classifier
        self.folds = folds
        self.classifiers = []
        self.gen_standardized()

    def gen_standardized(self):
        x = [d['x'] for d in self.data]
        y = [d['y'] for d in self.data]
        mean_train = np.mean(x, axis=0)
        mean_train[-1] = 0
        std_train = np.std(x, axis=0)
        std_train[-1] = 1
        std_train[std_train == 0] = 1
        std = (x - mean_train) / std_train
        min_train = np.min(x, axis=0)
        range_train = np.max(x, axis=0) - min_train
        range_train[range_train == 0] = 1
        minmax = (x - min_train) / range_train
        minmax.clip(0, 1)
        self.standardized = {}
        self.standardized['none'] = self.data
        self.standardized['std'] = np.array([{'x': std[i], 'y': y[i]}
                                             for i in range(len(self.data))])
        self.standardized['minmax'] = np.array(
            [{'x': minmax[i], 'y': y[i]} for i in range(len(self.data))])
        self.mean_train = mean_train
        self.std_train = std_train
        self.min_train = min_train
        self.range_train = range_train

    def standardize(self, x, std='std'):
        if std == 'std':
            x = (x - self.mean_train) / self.std_train
        elif std == 'minmax':
            x = (x - self.min_train) / self.range_train
            x.clip(0, 1)
        return x

    def get_classifier(self, classifier_name, train_instances, std, cl_args):
        if (std != 'std') and (std != 'minmax'):
            std = 'none'
        data = train_instances[std]
        classifier = gen_classifier(classifier_name, data, **cl_args)
        return {'classifier': classifier, 'std': std}

    def single_classify(self, classifier, x):
        c = classifier['classifier']
        s = classifier['std']
        x = self.standardize(x, s)
        return c.classify(x)

    def build(self):
        permutation = np.random.permutation(len(self.data))
        permuted = {}
        for k in self.standardized.keys():
            permuted[k] = self.standardized[k][permutation]
        fold_size = int(len(self.data) / self.folds)
        slices = {}
        for k in self.standardized.keys():
            slices[k] = [permuted[k][int(f * fold_size):int((f + 1) * fold_size)]
                         for f in range(self.folds)]
        meta_train = []
        for f in range(self.folds):
            train_instances = {}
            test_instances = {}
            for k in self.standardized.keys():
                train_instances[k] = np.concatenate(
                    [slices[k][i] if i != f else [] for i in range(len(slices[k]))])
                test_instances[k] = slices[k][f]
            classifiers = []
            for classifier in self.classifier_names:
                classifiers.append(self.get_classifier(
                    classifier['name'], train_instances, classifier.get('std', 'none'), classifier.get('args', {})))
            for i in range(len(test_instances['none'])):
                res = []
                conf = []
                for classifier in classifiers:
                    # add confidence?
                    cl = classifier['classifier']
                    st = classifier['std']
                    x = test_instances[st][i]['x']
                    r, c = cl.classify(x)
                    res.append(r)
                    conf.append(c)
                meta_train.append(
                    {'x': np.concatenate([res, conf]), 'y': test_instances['none'][i]['y']})
        classifier_name = self.meta_classifier_name['name']
        cl_args = self.meta_classifier_name['args']
        self.meta_model = self.get_classifier(
            classifier_name, {'none': meta_train}, 'none', cl_args)
        for classifier in self.classifier_names:
            self.classifiers.append(self.get_classifier(
                classifier['name'], self.standardized, classifier.get('std', 'none'), classifier.get('args', {})))

    def classify_all(self, x):
        res = []
        conf = []
        for i in range(len(self.classifiers)):
            r, c = self.single_classify(self.classifiers[i], x)
            res.append(r)
            conf.append(c)
        return np.array(res), np.array(conf)

    def classify(self, x):
        res, conf = self.classify_all(x)
        return self.single_classify(self.meta_model, np.concatenate([res, conf]))

