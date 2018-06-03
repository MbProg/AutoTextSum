from collections import namedtuple


class evaluate:


    def __init__(self,nugget_prediction, nugget_gold):
        self.nugget_prediction = nugget_prediction
        self.nugget_gold = nugget_gold


    def set_conf_matrix_values(self):
        '''
            Computes and returns parameters of confusion metric
        '''

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        predicted_nugget_list=[nugget for nugget in self.nugget_prediction]
        for nugget_label_turple in self.nugget_gold:
            if nugget_label_turple[0] in predicted_nugget_list:
                if nugget_label_turple[1] == 1:
                    true_pos +=1
                else:
                    false_pos +=1
            else :
                if nugget_label_turple[1]==1:
                    false_neg +=1
                else:
                    true_neg += 1
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

        metric = self.set_conf_matrix_values()
        return (metric.true_pos+metric.true_neg)/len(self.nugget_gold)

    def precision(self):
        '''
        Computes precision
        :return:precision
        '''

        metric = self.set_conf_matrix_values()
        return(metric.true_pos/(metric.true_pos+ metric.false_pos))

    def recall(self):
        '''
        computes recall
        :return:Recall
        '''
        metric = self.set_conf_matrix_values()
        return(metric.true_pos/metric.true_pos + metric.false_neg)

    def F1_score(self):
        '''
        Computes F1 Score
        :return:F1_score
        '''
        return(2 * self.recall()*self.precision())/(self.recall() + self.precision())








