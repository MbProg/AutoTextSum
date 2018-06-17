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
from keras.layers import Input, Dense, GRU, Embedding, LSTM, Dense, Dropout, Flatten, Bidirectional
from keras.models import Model

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
        self.model = self.build_network()
        # self.word2vec =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # self.true_nuggets = reader.nuggets

    def build_network(self, hidden_dim=256, GRU_dim=256):
        word_embedding_dim = 300
        sentence_embedding_dim = 512
        hidden_dim = hidden_dim
        GRU_dim = GRU_dim
        words_shape = (None, word_embedding_dim)
        query_shape = (None, word_embedding_dim)


        word_sequence = Input(shape=words_shape,name='inputwords')
        query = Input(shape=query_shape)
        sentence_embedding = Input(shape=(sentence_embedding_dim,),name='inputsent')
        # process the sentence embedding
        x1 = Dense(hidden_dim, activation='relu')(sentence_embedding)
        # weights for query and word sequence are shared
        gru_shared = GRU(GRU_dim, activation='relu')
        # process the word sequence
        x2 = gru_shared(word_sequence)
        # process the query
        x3 = gru_shared(query)

        merged_outputs = keras.layers.concatenate([x1, x2, x3], axis=-1)
        x1 = Dense(hidden_dim, activation='relu')(merged_outputs)
        # regression
        output = Dense(1, activation='linear')(x1)
        model = Model(inputs=[word_sequence, query, sentence_embedding], outputs=output)
        model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
        return model

    def pickle_generator(self, pathx = 'Data\Xtrain', pathy='Data/Ytrain', path_sent='Data/SentEmbeddings'):
        with open('Data/Xtrain', 'rb') as fx, open('Data/Ytrain', 'rb') as fy, \
                open('Data/nugget_candidates', 'rb') as fn, \
                open('Data/SentEmbeddings', 'rb') as fs, \
                open('Data/queries', 'rb') as fq, \
                open('Data/query_sent_embeddings', 'rb') as fqs:
            try:
                while True:
                    yield pickle.load(fx), pickle.load(fy), pickle.load(fs), pickle.load(fn), pickle.load(fq)
            except EOFError:
                raise StopIteration

    def preprocess(self, batch_size = 64, num_batches = np.inf):

        r = CorpusReader()
        feature_builder = SimpleFeatureBuilder(r, batch_size=batch_size, limit_embeddings=0)
        gen = feature_builder.generate_sequence_word_embeddings(max_len=6, seed=1)
        # preprocess word and sentence embeddings
        i=0
        with open('Data/Xtrain', 'wb') as fx, open('Data/Ytrain', 'wb') as fy, \
                open('Data/nugget_candidates','wb') as fn,\
                open('Data/SentEmbeddings', 'wb') as fs, \
                open('Data/queries', 'wb') as fq, \
                open('Data/query_sent_embeddings', 'wb') as fqs:
            while True:
                if i >= num_batches:
                    break
                # with open('Xtrain', mode='a+') as fileX, open('Ytrain', mode='a+') as fileY:
                try:
                    x, y, nugget_candidates, queries, query_sent_embeddings = next(gen)
                except StopIteration as e:
                    print('Iteration ended')
                    break
                sentence_embeddings = feature_builder.generate_sentence_embeddings(nugget_candidates, tokenized=True)
                assert sentence_embeddings.shape == (batch_size, 512), 'sentence embeddings are shape: {} \n ' \
                                                                       'for Embeddings of {}'.format(sentence_embeddings.shape, nugget_candidates)

                # append to each example
                # x = [embedded_seq + sentence_embeddings[i] for i, embedded_seq in enumerate(x)]
                pickle.dump(x, fx)
                pickle.dump(y, fy)
                pickle.dump(sentence_embeddings, fs)
                pickle.dump(nugget_candidates, fn)
                pickle.dump(queries, fq)
                pickle.dump(query_sent_embeddings, fqs)
                i += 1

if __name__ == '__main__':
    batch_size = 64
    n = Nugget_Classifier()
    n.preprocess(batch_size,)
    #train

    batch_generator = n.pickle_generator()
    #batch_words, batch_y, batch_sent, nugget, queries = next(batch_generator)
    #
    # model_words = keras.Sequential()
    # model_words.add(LSTM(100, input_shape=(None, 300)))
    # # model_words.add(Dense(100))
    # model_words.add(Dense(1))
    #
    # model_words.compile('rmsprop',
    #                     'mean_squared_error',
    #                     ['accuracy'])
    # model_words.train_on_batch(batch_words, batch_y)
    # model_words.summary()
    # print(model_words)

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
    epochs = 5
    # for i in range(epochs):
    #     try:
    #
    #     except StopIteration:
    #         break
    #     model_sent.train_on_batch(sent_batch, y_batch)
    # model_sent.evaluate(sent_batch, y_batch)
    # res = model_sent.evaluate(np.array(list(chain(*sentence_embeddings[:test_cutoff]))), np.array(list(chain(*Ytrain[:test_cutoff]))))
    names = model_sent.metrics_names
    # print('Result: {} {}'.format(names, res))
    # print(model_sent.predict(sentence_embeddings[0]), '\nTrue: {}'.format(Ytrain[0]))
    ### Testing the combined model with random Data
    x_batch, y_batch, sent_batch, nuggets, queries = next(batch_generator)
    test_sentences = np.random.randn(64, 15, 300)
    test_query = np.random.randn(64, 4, 300)
    sentence_embedding = np.random.randn(64, 512)
    test_labels = np.random.randn(64)
    print(type(nuggets), type(queries), type(sentence_embedding))
    X = [np.array(x_batch), test_query, sentence_embedding]
    n.model.train_on_batch(X, y_batch)
    print(n.model.predict(X))