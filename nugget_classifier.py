from corpus_reader import  CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
import gensim
import pandas as pd
from pprint import pprint
from nltk import word_tokenize
from nltk import sent_tokenize
import numpy as np
import os
import pickle
from itertools import chain

import keras
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Bidirectional

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

    def pickle_generator(self, pathx = 'Data\Xtrain', pathy='Data/Ytrain', path_sent='Data/SentEmbeddings'):
        while True:
            try:
                with open('Data/Xtrain', 'wb') as fx, open('Data/Ytrain', 'wb') as fy, open('Data/SentEmbeddings', 'wb') as fs:
                    yield pickle.load(fx), pickle.load(fy), pickle.load(fs)
            except EOFError:
                raise StopIteration

    def preprocess(self, batch_size = 64, num_batches = np.inf):

        r = CorpusReader()
        feature_builder = SimpleFeatureBuilder(r, batch_size=batch_size, limit_embeddings=10000)
        gen = feature_builder.generate_sequence_word_embeddings(max_len=6, seed=1)
        # preprocess word and sentence embeddings
        if not os.path.exists('Data/Xtrain'):
            #todo
            Xtrain, Ytrain, sent_embeddings_train, nuggets = [], [], [], []
            while True:
                if len(Xtrain) > num_batches:
                    break
                # with open('Xtrain', mode='a+') as fileX, open('Ytrain', mode='a+') as fileY:
                try:
                    x, y, nugget_candidates = next(gen)
                except StopIteration as e:
                    print('Iteration ended')
                    break
                sentence_embeddings = feature_builder.generate_sentence_embeddings(nugget_candidates, tokenized=True)
                assert sentence_embeddings.shape == (batch_size, 512), 'sentence embeddings are shape: {} \n ' \
                                                                       'for Embeddings of {}'.format(sentence_embeddings.shape, nugget_candidates)

                # append to each example
                # x = [embedded_seq + sentence_embeddings[i] for i, embedded_seq in enumerate(x)]
                Xtrain.append(x)
                Ytrain.append(y)
                sent_embeddings_train.append(sentence_embeddings)
                print(len(Xtrain))
            with open('Data/Xtrain', 'wb') as fx, open('Data/Ytrain', 'wb') as fy, \
                    open('Data/SentEmbeddings', 'wb') as fs:
                pickle.dump(Xtrain, fx)
                pickle.dump(Ytrain, fy)
                pickle.dump(sent_embeddings_train, fs)

    # def train(self, ):
#TODO: Was ist 'I&#65533;&#65533;m???
#todo query abspeichern

if __name__ == '__main__':
    batch_size = 64
    n = Nugget_Classifier()
    n.preprocess(64, 7)
    #train

    batch_generator = n.pickle_generator()
    batch_words, batch_y, batch_sent = next(batch_generator)

    model_words = keras.Sequential()
    model_words.add(LSTM(100, input_shape=(None, 300)))
    # model_words.add(Dense(100))
    model_words.add(Dense(1))

    model_words.compile('rmsprop',
                        'mean_squared_error',
                        ['accuracy'])
    model_words.train_on_batch(batch_words, batch_y)
    model_words.summary()
    print(model_words)

    model_sent = keras.Sequential()
    depth, width = 2, 40
    test_cutoff = 2
    model_sent.add(Dense(width, input_shape=(512,)))
    # model_sent.add(Flatten())
    for i in range(1, depth):
        model_sent.add(Dense(width))
    model_sent.add(Dense(1))

    model_sent.summary()
    model_sent.compile('rmsprop',
                       'mean_squared_error',
                       ['accuracy'])
    for _, y_batch, sent_batch in batch_generator:
        model_sent.train_on_batch(sent_batch, y_batch)
    # res = model_sent.evaluate(np.array(list(chain(*sentence_embeddings[:test_cutoff]))), np.array(list(chain(*Ytrain[:test_cutoff]))))
    names = model_sent.metrics_names
    # print('Result: {} {}'.format(names, res))
    # print(model_sent.predict(sentence_embeddings[0]), '\nTrue: {}'.format(Ytrain[0]))