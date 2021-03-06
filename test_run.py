from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector

from simple_evaluator import Evaluate
import time

reader = CorpusReader(approach='word')
fb = SimpleFeatureBuilder(reader, word_vectors_path='../../GoogleNews-vectors-negative300.bin')
#Check if the reader correctly read the input
print(reader.nuggets.keys())
print(reader.paragraphs.keys())
print(reader.topics.ix[0].topic)
#initialize the model and train it batchwise with logistic regression
# nugget_detector = SimpleNuggetDetector(reader, fb, model='lr', params=None)
# nugget_detector.fit_model()
# nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()
#test the decision tree training with the whole dataset
params = {}
params['max_depth'] = 8
params['min_samples_split'] = 2
params['min_samples_leaf'] = 3
params['min_weight_fraction_leaf'] = 0.0
nugget_detector = SimpleNuggetDetector(reader, fb, model='tree', params=params)
nugget_detector.fit_model()
nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()

# nugget_predictions = [['Hello','I','am'],['I','go']]
# nugget_gold = [(['I','go'],1),(['This','is'],1),(['Hello','go'],0)]
timeBeforeEval = time.time()
objEvaluator = Evaluate(nugget_predictions,nugget_gold)
print('Accuracy :{}'.format(objEvaluator.accuracy()))
print('Recal:{}'.format(objEvaluator.recall()))
print('Precision:{}'.format(objEvaluator.precision()))
print('F1 Score:{}'.format(objEvaluator.F1_score()))
print('Time:',time.time() - timeBeforeEval)
