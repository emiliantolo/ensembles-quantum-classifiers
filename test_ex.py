import numpy as np
import pandas as pd
import os
import sys
import time
from classification.ensembles.bootstrap import BootstrapClassifier
from classification.ensembles.boosting import BoostingClassifier
from classification.ensembles.stacking import StackingClassifier
from classification.helpers.factories import gen_classifier
from classification.helpers.data import get_train_test

def test(test_data, classifier):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, t in enumerate(test_data):
        res, conf = classifier.classify(t['x'])
        print('Completed {}/{}'.format(i + 1, len(test_data)))
        if (res > 0) and (t['y'] > 0):
            tp += 1
        if (res > 0) and (t['y'] < 0):
            fp += 1
        if (res < 0) and (t['y'] > 0):
            fn += 1
        if (res < 0) and (t['y'] < 0):
            tn += 1
    if tp == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc, f1


def build_classifier(config, data):
    ensemble = config.get('ensemble')
    classifier_name = config.get('classifier')
    cl_args = config.get('args', {})
    if ensemble == 'single':
        classifier = gen_classifier(classifier_name, data, **cl_args)
    elif ensemble == 'bootstrap':
        classifier = BootstrapClassifier(
            data, classifier_name, **cl_args)
        classifier.build()
    elif ensemble == 'boosting':
        classifier = BoostingClassifier(
            data, classifier_name, **cl_args)
        classifier.build()
    elif ensemble == 'stacking':
        if classifier_name == 'quantum_combined':
            execution = cl_args.get('execution')
            shots = cl_args.get('shots')
            classifiers = [
                {'name': 'quantum_cosine', 'std': 'std', 'args': cl_args},
                {'name': 'quantum_distance', 'std': 'std', 'args': cl_args},
                {'name': 'quantum_kNN', 'std': 'minmax', 'args': {
                    'k': 1, 'execution': execution, 'shots': shots}},
                {'name': 'quantum_kNN', 'std': 'minmax', 'args': {
                    'k': 3, 'execution': execution, 'shots': shots}}]
            meta_classifier = {'name': 'quantum_kNN', 'args': {
                'k': 5, 'execution': execution, 'shots': shots}}
        elif classifier_name == 'combined':
            classifiers = [
                {'name': 'cosine', 'std': 'std'},
                {'name': 'distance', 'std': 'std'},
                {'name': 'kNN_cosine_squared', 'std': 'minmax', 'args': {'k': 1}},
                {'name': 'kNN_cosine_squared', 'std': 'minmax', 'args': {'k': 3}}]
            meta_classifier = {'name': 'quantum_kNN', 'args': {'k': 5}}
        else:
            raise ValueError('classifier_name not recognized')
        classifier = StackingClassifier(
            data, classifiers, meta_classifier)
        classifier.build()
    else:
        raise ValueError('classifier_name not recognized')
    return classifier

def parse_args():
    data_line = None
    fold_line = None
    classifier_line = None

    if len(sys.argv) > 1:
        line = sys.argv[1:]
        for i, l in enumerate(line):
            if (l == '--data') or (l == '-d'):
                data_line = line[i + 1]
            if (l == '--fold') or (l == '-f'):
                fold_line = line[i + 1]
            if (l == '--classifier') or (l == '-c'):
                classifier_line = line[i + 1]

    if data_line:
        s = data_line.split(':')
        if len(s) == 2:
            data_s = int(s[0]) if s[0] != '' else None
            data_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            data_s = int(s[0])
            data_e = data_s + 1
        else:
            data_s = None
            data_e = None
    else:
        data_s = None
        data_e = None

    if fold_line:
        s = fold_line.split(':')
        if len(s) == 2:
            fold_s = int(s[0]) if s[0] != '' else None
            fold_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            fold_s = int(s[0])
            fold_e = fold_s + 1
        else:
            fold_s = None
            fold_e = None
    else:
        fold_s = None
        fold_e = None

    if classifier_line:
        s = classifier_line.split(':')
        if len(s) == 2:
            classifier_s = int(s[0]) if s[0] != '' else None
            classifier_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            classifier_s = int(s[0])
            classifier_e = classifier_s + 1
        else:
            classifier_s = None
            classifier_e = None
    else:
        classifier_s = None
        classifier_e = None

    return data_s, data_e, fold_s, fold_e, classifier_s, classifier_e


# usage
# python test_ex.py
# python test_ex.py --data 0:1 --fold 0:1 --classifier 0:1
# python test_ex.py -d 0:1
# python test_ex.py -f :1
# python test_ex.py -c 0:
# python test_ex.py -d 0

if __name__ == '__main__':

    np.random.seed(123)

    # args
    data_s, data_e, fold_s, fold_e, classifier_s, classifier_e = parse_args()

    # folds
    n_rep = 10
    folds_to_execute = list(range(n_rep))

    # data
    dir = os.listdir('data/dataset')
    dir.remove('02_transfusion.csv')
    dir.sort()
    test_size = 0.2
    stds = ['none', 'std', 'minmax']

    if not os.path.exists('results'):
        os.makedirs('results')

    # classifiers
    samples = 8
    N = 30
    shots = 8192
    classifiers = [
        #single
        {'ensemble': 'single', 'classifier': 'distance'},
        {'ensemble': 'single', 'classifier': 'quantum_distance',
        'args': {'execution': 'statevector'}},
        {'ensemble': 'single', 'classifier': 'quantum_distance',
        'args': {'shots': shots, 'execution': 'local'}},
        #bootstrap
        {'ensemble': 'bootstrap', 'classifier': 'quantum_distance',
        'args': {'N': N, 'n_samples': samples, 'shots': shots, 'execution': 'local'}},
        {'ensemble': 'bootstrap', 'classifier': 'quantum_distance',
        'args': {'N': N, 'n_samples': samples, 'shots': shots, 'execution': 'local', 'balanced': True}},
        #boosting
        {'ensemble': 'boosting', 'classifier': 'quantum_distance',
        'args': {'N': N, 'n_samples': samples, 'shots': shots, 'execution': 'local'}},
        {'ensemble': 'boosting', 'classifier': 'quantum_distance',
        'args': {'N': N, 'n_samples': samples, 'shots': shots, 'execution': 'local', 'balanced': True}},
        #stacking
        {'ensemble': 'stacking', 'classifier': 'quantum_combined',
        'args': {'shots': shots, 'execution': 'local'}}
    ]

    # select subset
    dir = dir[data_s:data_e]
    folds_to_execute = folds_to_execute[fold_s:fold_e]
    classifiers = classifiers[classifier_s:classifier_e]

    log = 'results-{}-{}.csv'.format(
        os.path.basename(sys.argv[0]).split('.')[0], time.time())
    print('Running: {}'.format(log))
    print('Data:\n{}'.format('\n'.join(dir)))
    print('Folds: {}'.format(folds_to_execute))

    results = []
    for i in folds_to_execute:
        for d in dir:
            f = 'data/folds/{}/{}.txt'.format(d.split('.')[0], str(i))
            fold = np.genfromtxt(f)
            print('\nFold: {}'.format(f))
            for std in stds:
                train_data, test_data = get_train_test(fold, test_size, std=std)
                for c in classifiers:
                    ensemble = c.get('ensemble')
                    classifier_name = c.get('classifier')
                    args = c.get('args', {})
                    start = time.time()
                    print('Started at: {} \t {} - {} - {} - {}'.format(time.ctime(),
                        ensemble, classifier_name, std, args))
                    classifier = build_classifier(c, train_data)
                    print('Classifier built, testing...')
                    acc, f1 = test(test_data, classifier)
                    res = {'fold': f, 'standardization': std, 'ensemble': ensemble,
                        'classifier': classifier_name, 'acc': acc, 'f1': f1}
                    res |= args
                    results.append(res)
                    print('Finished at: {} \t Time: {}'.format(
                        time.ctime(), (time.time() - start)))
                    pd.DataFrame(results).to_csv(
                        'results/{}'.format(log), sep=";", decimal=",")
