from collections import namedtuple
import time

class Evaluate:


    def __init__(self,nugget_prediction, nugget_gold):
        self.nugget_prediction = nugget_prediction
        self.nugget_gold = nugget_gold
        self.predictions = {}
        self.prepare_data()
        self.metric = self.set_conf_matrix_values()

    def prepare_data(self):
        for nugget in self.nugget_prediction:
            self.predictions[repr(nugget)] = 1
        

    def set_conf_matrix_values(self):
        '''
            Computes and returns parameters of confusion metric
        '''
        time.sleep(5)
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        # predicted_nugget_list=[nugget for nugget in self.nugget_prediction]

        for nugget,label in self.nugget_gold:
            if repr(nugget) in self.predictions:
                if label == 1:
                    true_pos +=1
                else:
                    false_pos +=1
            else: 
                if label ==1:
                    false_neg += 1
                else:
                    true_neg += 1
                
        # nugget_gold: all nuggets labeled as 0 and 1
        # nugget_prediction: only nuggets that was classified as 1
        # for nugget_label_turple in self.nugget_gold:
        #     if nugget_label_turple[0] in self.nugget_prediction:
        #         if nugget_label_turple[1] == 1:
        #             true_pos +=1
        #         else:
        #             false_pos +=1
        #     else :
        #         if nugget_label_turple[1]==1:
        #             false_neg +=1
        #         else:
        #             true_neg += 1
        metric = namedtuple('metric',['true_pos','true_neg','false_pos','false_neg'])
        metric.false_neg = false_neg
        metric.false_pos=false_pos
        metric.true_neg = true_neg
        metric.true_pos = true_pos

        return metric


    def accuracy(self):
        '''
        Computes accuracy
        :return:Accuracy
        '''

        return float(self.metric.true_pos+self.metric.true_neg)/len(self.nugget_gold)

    def precision(self):
        '''
        Computes precision
        :return:precision
        '''

        return(self.metric.true_pos/(self.metric.true_pos+ self.metric.false_pos))

    def recall(self):
        '''
        computes recall
        :return:Recall
        '''
        return(self.metric.true_pos/(self.metric.true_pos + self.metric.false_neg))

    def F1_score(self):
        '''
        Computes F1 Score
        :return:F1_score
        '''
        return(2 * self.recall()*self.precision())/(self.recall() + self.precision())








