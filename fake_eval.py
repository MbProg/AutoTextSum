from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector

from simple_evaluator import Evaluate
import time
import numpy as np
params = {}
params['max_depth'] = 12
params['min_samples_split'] = 2
params['min_samples_leaf'] = 3
params['min_weight_fraction_leaf'] = 0

X = np.load('X_sample.npy')
y = np.load('y_sample.npy')
nugget_detector = SimpleNuggetDetector(None, None, model='tree', params=params)
# just make sure the model was initialized properly and can be fitted
nugget_detector.fit_model(X=X, y=y)


f = open('gold_nuggets', 'r')
nugget_gold = eval(f.read())
f.close()
# make random predictions
nugget_predictions, nugget_gold = nugget_detector.fake_predict(nugget_gold)

objEvaluator = Evaluate(nugget_predictions,nugget_gold)
print('Accuracy :{}'.format(objEvaluator.accuracy()))
print('Recal:{}'.format(objEvaluator.recall()))
print('Precision:{}'.format(objEvaluator.precision()))
print('F1 Score:{}'.format(objEvaluator.F1_score()))
