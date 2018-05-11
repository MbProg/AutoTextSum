from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from simple_feature_builder import SimpleFeatureBuilder
from sklearn.linear_model import SGDClassifier
import numpy as np

class SimpleNuggetDetector:

    def __init__(self, corpus_reader, feature_builder, model='tree', params=None):
        self.feature_builder = feature_builder
        self.corpus_reader = corpus_reader
        if params and model == 'tree':
            self.model = DecisionTreeClassifier(max_depth=params['max_depth'],
                                                min_samples_split=params['min_samples_split'],
                                                min_samples_leaf=params['min_samples_leaf'],
                                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'])
        elif not params and model == 'tree':
            self.model = DecisionTreeClassifier()
        else:
            # default logistic regression
            self.model = SGDClassifier(loss='log', n_jobs=2)

    def fit_model(self):
        training_iterator = self.feature_builder.get_word_embedding_features()
        classes = np.array([i for i in range(self.feature_builder.corpus_reader.max_occurrence)])
        first_fit = True
        for X_batch, y_batch in training_iterator:
            if first_fit:
                self.model.partial_fit(X_batch, y_batch, classes=classes)
                first_fit = False
            else:
                self.model.partial_fit(X_batch, y_batch)
