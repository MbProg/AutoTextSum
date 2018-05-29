from corpus_reader import  CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
import gensim
import pandas as pd
from pprint import pprint
from nltk import word_tokenize
from nltk import sent_tokenize

import keras
from keras.layers import Embedding, LSTM, Dense, Dropout

class Nugget_Classifier():
    '''
    Second Approach: Define a maximum nugget length of
    a. Then for each sentence in a paragraph form all nuggets that have a length
    of <= a. The labels could either be 0/1 or a multiclass for the amount of
    workers that actually picked the nugget. The feature representation of the
    nuggets/query could again be word vectors, or a bag of words representation,
    bigrams, etc.. For prediction a new sentence would then be split up again in
    nuggets of length <= a and then brought to the corresponding feature representation. Afterwards there could be a ranking according to the predicted
    probabilities of the nuggets and then the highest ranking non overlapping
    nuggets that achieve a certain threshold of probability would be chosen.

    Args:
            limit_embeddings (int): limit loaded embeddings,
            to speed up loading and debugging. 0 equals no limit
    '''
    def __init__(self, reader = None, limit_embeddings = 0):
        if reader is None:
            self.reader = CorpusReader()
        else:
            self.reader = reader
        # self.word2vec =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # self.true_nuggets = reader.nuggets

#TODO: 'I&#65533;&#65533;m???

if __name__ == '__main__':
    r = CorpusReader()
    feature_builder = SimpleFeatureBuilder(r, batch_size= 64, limit_embeddings=50000)
    gen = feature_builder.generate_sequence_word_embeddings(max_len=3, seed=1)
    print(next(gen))

