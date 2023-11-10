from classification.classifiers.classical import CosineClassifier, DistanceClassifier, KNNCosineSquaredClassifier
from classification.classifiers.quantum import QuantumCosineClassifier, QuantumDistanceClassifier, QuantumKNNClassifier


def gen_classifier(classifier_name, data, **cl_args):
    if classifier_name == 'cosine':
        classifier = CosineClassifier(data, **cl_args)
    elif classifier_name == 'distance':
        classifier = DistanceClassifier(data, **cl_args)
    elif classifier_name == 'kNN_cosine_squared':
        classifier = KNNCosineSquaredClassifier(data, **cl_args)
    elif classifier_name == 'quantum_cosine':
        classifier = QuantumCosineClassifier(data, **cl_args)
    elif classifier_name == 'quantum_distance':
        classifier = QuantumDistanceClassifier(data, **cl_args)
    elif classifier_name == 'quantum_kNN':
        classifier = QuantumKNNClassifier(data, **cl_args)
    else:
        raise ValueError('classifier_name not recognized')
    return classifier
