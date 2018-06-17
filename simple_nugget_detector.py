from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from simple_feature_builder import SimpleFeatureBuilder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from functools import reduce
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
class SimpleNuggetDetector:

    def __init__(self, corpus_reader, feature_builder, model='tree', params=None):
        self.feature_builder = feature_builder
        self.score_threshold = 2
        if params and model == 'tree':
            self.model = DecisionTreeClassifier(max_depth=params['max_depth'],
                                                min_samples_split=params['min_samples_split'],
                                                min_samples_leaf=params['min_samples_leaf'],
                                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'])
            self.train_mode = 'full_mode'
        elif params and model == 'svm':
            self.model = SVC(C=params['C'], kernel='linear', probability=True)
            self.train_mode = 'full_mode'
        elif params and model == 'random_forest':
            self.model = RandomForestClassifier(max_depth=params['max_depth'],
                                                n_estimators=params['n_estimators'],
                                                min_samples_split=params['min_samples_split'],
                                                min_samples_leaf=params['min_samples_leaf'],
                                                min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                                n_jobs=-1,
                                                )
            self.train_mode = 'full_mode'
        else:
            # default logistic regression
            self.model = SGDClassifier(loss='log', n_jobs=2)
            self.train_mode = 'batch_mode'

    def fit_model(self):
        if self.train_mode == 'batch_mode':
            training_iterator = self.feature_builder.get_word_embedding_features()
            classes = np.array([i for i in range(self.feature_builder.corpus_reader.max_occurrence)])
            first_fit = True
            batch_count = 0
            for X_batch, y_batch in training_iterator:
                print(batch_count)
                if first_fit:
                    # have to tell the model how many classes exist when we learn on the first batch
                    self.model.partial_fit(X_batch, y_batch, classes=classes)
                    first_fit = False
                else:
                    self.model.partial_fit(X_batch, y_batch)
                    batch_count += 1
                # only to reduce training time for testing, delete this for training the final model
                if batch_count == 200:
                    break
        else:
            X, y = self.feature_builder.get_complete_word_embedding_features()
            # iterator will yield just one batch of all examples
            self.model.fit(X, y)

    def convert_word_to_nugget_predictions(self, sentences, decision='binary'):
        nugget_predictions = []
        if decision=='binary':
            # if decision is binary just return the binary value
            include_word = lambda x1, x2: x1
        else:
            include_word = lambda x1, x2: x1 >= x2
        for sentence in sentences:
            nugget = []
            current_nugget = False
            for word,score in sentence:
                if include_word(score, self.score_threshold):
                    nugget.append(word)
                    current_nugget = True
                elif not include_word(score, self.score_threshold) and current_nugget:
                    nugget_predictions.append(nugget)
                    nugget = []
                    current_nugget = False
                else:
                    pass
        return nugget_predictions

    def convert_predictions(self, predictions):
        ''' Convert predictions for all classes to a binary decision by summing probabilities
        up to the threshold for the negative class and above for the positive class label.
        '''
        summed_prob_negative = np.sum(predictions[:,:self.score_threshold],1)
        summed_prob_positive = np.sum(predictions[:,self.score_threshold:],1)
        return summed_prob_positive > summed_prob_negative

    def predict_dev_set(self):
        ''' Make predictions on the dev set that is stored in the corpus reader
        and concatenate them into nuggets according to their predicted score. Returns a list of the predicted nuggets and a list of all possible nuggets
        with a label that signals if that possible nugget also was chosen by enough workers.
        '''
        dev_set_iterator = self.feature_builder.get_word_embedding_features(mode='evaluate')
        predictions = []
        words = []
        labels = []
        sentences = []
        sentence = []
        for X_batch, y_batch, word, count, end_sentence in dev_set_iterator:
            prediction = self.model.predict_proba(X_batch)
            prediction = self.convert_predictions(prediction)
            predictions.append(prediction)
            sentence.append((word, prediction))
            words.append(word)
            labels.append(count)
            if end_sentence:
                sentences.append(sentence)
                sentence = []

        dev_set_text_ids =  [self.feature_builder.corpus_reader.topics.ix[x].text_id for x in self.feature_builder.corpus_reader.devset_topics]
        #nugget_gold = [self.feature_builder.corpus_reader.nuggets[str(text_id)] for text_id in dev_set_text_ids]
        # build the predicted nuggets from single word predictions
        nugget_predictions = self.convert_word_to_nugget_predictions(sentences)
        # get the gold nuggets from the labeled data
        nuggets_gold = []
        for dev_set_id in dev_set_text_ids:
            #print(self.feature_builder.corpus_reader.get_paragraph_nugget_pairs(dev_set_id), dev_set_id)
            paragraph_nugget_tuples = self.feature_builder.corpus_reader.get_paragraph_nugget_pairs(str(dev_set_id))
            for paragraph, paragraph_nuggets in paragraph_nugget_tuples:
                all_nuggets = self.feature_builder.__get_potential_nuggets__(paragraph, max_len=10)
                nugget_gold = [(nugget, paragraph_nuggets.get(' '.join([w for w in nugget]), 0)) for nugget in all_nuggets if nugget]
                nugget_gold = [(nugget, 1) if score>= self.score_threshold else (nugget, 0) for nugget, score in nugget_gold]
                nuggets_gold += nugget_gold
        return nugget_predictions, nuggets_gold
