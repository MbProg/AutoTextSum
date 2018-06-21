from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector
from simple_evaluator import evaluate
import random as rd
import numpy as np
import time
import json


reader = CorpusReader(approach='word')
fb = SimpleFeatureBuilder(reader, word_vectors_path='../GoogleNews-vectors-negative300.bin')
x= np.load('/home/tekams/PycharmProjects/AutoTextSum-master/textSUmmarizer/X_sample.npy')
y = np.load('/home/tekams/PycharmProjects/AutoTextSum-master/textSUmmarizer/y_sample.npy')
f = open('/home/tekams/PycharmProjects/AutoTextSum-master/textSUmmarizer/gold_nuggets', 'r')
nugget_gold = eval(f.read())
f.close()
with open('tree.txt', 'w') as file:
    for i in range(2):
        eval={}
        params = {}
        params['max_depth'] = rd.randint(3, 10)
        params['min_samples_split'] = rd.randint(2, 10)
        params['min_samples_leaf'] = rd.randint(5, 10)
        params['min_weight_fraction_leaf'] = 0.0
        print(params)

        nugget_detector = SimpleNuggetDetector(reader, fb, model='tree', params=params)
        nugget_detector.fit_model(x, y)
        nugget_predictions, nugget_gold = nugget_detector.fake_predict(nugget_gold)

        objEvaluator = evaluate(nugget_predictions, nugget_gold)
        print('Accuracy :{}'.format(objEvaluator.accuracy()))
        print('Recall:{}'.format(objEvaluator.recall()))
        print('Precision:{}'.format(objEvaluator.precision()))
        print('F1 Score:{}'.format(objEvaluator.F1_score()))

        eval['Accuracy'] = objEvaluator.accuracy()
        eval['Recall'] =  objEvaluator.recall()
        eval['Precision'] = objEvaluator.precision()
        eval['F1_Score']= objEvaluator.F1_score()

        file.write('iteration %s \n' % i, )
        file.write(json.dumps(eval))
        file.write('\n')
        file.write(params)
        file.write('\n')



# with open('svm.txt', 'w') as file:
#
#     for i in range(10):
#         eval = {}
#         params = {}
#         params['C'] = rd.randint(2, 10)
#         nugget_detector = SimpleNuggetDetector(reader, fb, model='svm', params=params)
#         nugget_detector.fit_model()
#         nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()
#
#         objEvaluator = evaluate(nugget_predictions, nugget_gold)
#         print('Accuracy :{}'.format(objEvaluator.accuracy()))
#         print('Recall:{}'.format(objEvaluator.recall()))
#         print('Precision:{}'.format(objEvaluator.precision()))
#         print('F1 Score:{}'.format(objEvaluator.F1_score()))
#
#         eval['Accuracy'] = objEvaluator.accuracy()
#         eval['Recall'] = objEvaluator.recall()
#         eval['Precision'] = objEvaluator.precision()
#         eval['F1_Score'] = objEvaluator.F1_score()
#
#         file.write('iteration %s \n' % i, )
#         file.write(json.dumps(eval))
#         file.write('\n')
#         file.write(params)
#         file.write('\n')
#
# with open('random_forest.txt', 'w') as file:
#     for i in range(10):
#         eval={}
#         params = {}
#         params['max_depth'] = rd.randint(3, 10)
#         params['n_estimators'] = rd.randint(3, 10)
#         params['min_samples_split'] = rd.randint(2, 10)
#         params['min_samples_leaf'] = rd.randint(5, 10)
#         params['min_weight_fraction_leaf'] = 0.0
#         print(params)
#
#         nugget_detector = SimpleNuggetDetector(reader, fb, model='tree', params=params)
#         nugget_detector.fit_model()
#         nugget_predictions, nugget_gold = nugget_detector.predict_dev_set()
#
#         objEvaluator = evaluate(nugget_predictions, nugget_gold)
#         print('Accuracy :{}'.format(objEvaluator.accuracy()))
#         print('Recall:{}'.format(objEvaluator.recall()))
#         print('Precision:{}'.format(objEvaluator.precision()))
#         print('F1 Score:{}'.format(objEvaluator.F1_score()))
#
#         eval['Accuracy'] = objEvaluator.accuracy()
#         eval['Recall'] =  objEvaluator.recall()
#         eval['Precision'] = objEvaluator.precision()
#         eval['F1_Score']= objEvaluator.F1_score()
#
#         file.write('iteration %s \n' % i, )
#         file.write(json.dumps(eval))
#         file.write('\n')
#         file.write(params)
#         file.write('\n')