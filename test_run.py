from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector
reader = CorpusReader(approach='word')
fb = SimpleFeatureBuilder(reader, word_vectors_path='../../GoogleNews-vectors-negative300.bin')
#Check if the reader correctly read the input
print(reader.nuggets.keys())
print(reader.paragraphs.keys())
print(reader.topics.ix[0].topic)
#initialize the model and train it batchwise with logistic regression
nugget_detector = SimpleNuggetDetector(reader, fb, model='lr', params=None)
nugget_detector.fit_model()
nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()
#test the decision tree training with the whole dataset
params = {}
params['max_depth'] = 8
params['min_samples_split'] = 2
params['min_samples_leaf'] = 3
params['min_weight_fraction_leaf'] = 0.0
nugget_detector = SimpleNuggetDetector(reader, fb, model='tree', params=params)
nugget_detector.fit_model()
nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()
