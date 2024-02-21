import numpy as np
import pandas as pd
import os
import sys
import time
from classification.helpers.data import get_train_test
from helpers import *

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
        {'ensemble': 'single', 'classifier': 'quantum_distance',
        'args': {'shots': shots, 'execution': 'local+noise'}},
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
