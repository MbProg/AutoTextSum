from corpus_reader import  CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
import gensim
import pandas as pd
from pprint import pprint
from nltk import word_tokenize
from nltk import sent_tokenize
import numpy as np
import os
import time
import pickle
from itertools import chain
import h5py

import keras
from keras.layers import Input, Dense, GRU, Embedding, LSTM, Dense, Dropout, Flatten, Bidirectional
from keras.models import Model
import tensorflow as tf

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

    def build_network(self, hidden_dim=256, GRU_dim=256, nb_words=25000, max_sequence_length=30, we_matrix=None):
        word_embedding_dim = 300
        hidden_dim = hidden_dim
        GRU_dim = GRU_dim


        word_sequence = Input(shape=(max_sequence_length,),name='inputwords')
        query = Input(shape=(max_sequence_length,))
        wv_layer = Embedding(nb_words,
                             word_embedding_dim,
                             mask_zero=False,
                             weights=[we_matrix],
                             input_length=max_sequence_length,
                             trainable=False)
        word_sequence_embedding = wv_layer(word_sequence)
        query_embedding = wv_layer(query)
        # weights for query and word sequence are shared
        gru_shared = GRU(GRU_dim, activation='relu')
        # process the word sequence
        x2 = gru_shared(word_sequence_embedding)
        # process the query
        x3 = gru_shared(query_embedding)

        merged_outputs = keras.layers.concatenate([x2, x3], axis=-1)
        x1 = Dense(hidden_dim, activation='relu')(merged_outputs)
        # regression
        output = Dense(1, activation='linear')(x1)
        model = Model(inputs=[word_sequence, query], outputs=output)
        model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
        self.model = model

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

        # preprocess word and sentence embeddings
        i=0
        feature_builder = SimpleFeatureBuilder(self.reader, batch_size=batch_size, limit_embeddings=0)
        gen = feature_builder.generate_sequence_word_embeddings(max_len=8, seed=1)
        #append mode!
        first_write = True
        while True:
            #todo hdf5 file metadata
            #roup_i = h5.create_group('batch{}'.format(i))
            print(time.strftime("%H:%M:%S")+': Batch {}'.format(i))
            if i >= num_batches:
                break
            try:
                x, y, nuggets, queries, query_embeddings = next(gen)
            except StopIteration as e:
                print('Iteration ended')
                break


            #nuggets_sentence_embeddings = feature_builder.generate_sentence_embeddings(nuggets, tokenized=True)
            #assert nuggets_sentence_embeddings.shape == (batch_size, 512), 'sentence embeddings are shape: {} \n ' \
            #                                                           'for Embeddings of {}'.format(nuggets_sentence_embeddings.shape, nuggets)

            # append to each example
            # x = [embedded_seq + sentence_embeddings[i] for i, embedded_seq in enumerate(x)]
            #print(nuggets)
            print(x.shape,x.dtype)
            print(y.shape,y.dtype)
            np.save('Data/word_sequence/x_{}'.format(i), x)
            np.save('Data/labels/y_{}'.format(i), y)
            #np.save('Data/sent_embedding/sent_{}'.format(i), nuggets_sentence_embeddings)
            f = open('Data/queries/query_{}'.format(i), 'w')
            f.write(repr(queries))
            f.close()
            f = open('Data/nuggets/nuggets_{}'.format(i), 'w')
            f.write(repr(nuggets))
            f.close()
            #del x, y, nuggets, queries, query_sent_embeddings

            i += 1

if __name__ == '__main__':
    batch_size = 64
    n = Nugget_Classifier()
    n.preprocess(batch_size,)
    #train
    '''
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
    '''
