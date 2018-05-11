import gensim
from nltk import word_tokenize
import numpy as np

class SimpleFeatureBuilder:

    def __init__(self, corpus_reader, train_size=0.8, batch_size=64, embedding_dim=300):
        self.word2vec =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.corpus_reader = corpus_reader
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def get_word2vec(self, word):
        try:
            return self.word2vec[word]
        except:
            #if the word does not exist return a zero vector
            return np.zeros(self.embedding_dim)

    def get_word_embedding_features(self):
        '''
        Returns training instances iteratively that can be build by iterating through the corpus. Each instance is the average word embedding of a given
        word in a sentence and its two surrounding words and the average of the topic/query words. The labels consist of the count of how many workers did choose to put that word in a
        nugget.
        '''
        paragraph_word_scores = self.corpus_reader.data
        # Initialize the features and labels
        X_batch,y_batch = ([], [])
        for topic_index, paragraph in paragraph_word_scores:
            topic_words = word_tokenize(self.corpus_reader.topics.ix[topic_index].topic)
            topic_word_embeddings = np.average([self.get_word2vec(word) for word in topic_words],0)
            for sentence_word_occurrences in paragraph:
                #print(sentence_word_occurrences)
                for i, (word, count) in enumerate(sentence_word_occurrences):
                    # get the current plus surrounding words of the index
                    if i > 0 and i < len(sentence_word_occurrences)-1:
                        surrounding_words = [sentence_word_occurrences[i-1][0]] + [word] + [sentence_word_occurrences[i+1][0]]
                    elif i==0:
                        surrounding_words = [word] + [sentence_word_occurrences[i+1][0]]
                    else:
                        surrounding_words = [sentence_word_occurrences[i-1][0]] + [word]

                    surrounding_word_embeddings = np.average([self.get_word2vec(word) for word in surrounding_words],0)
                    X_batch.append(np.average([topic_word_embeddings, surrounding_word_embeddings],0))
                    y_batch.append(count)
                    if len(X_batch) == self.batch_size:
                        yield np.array(X_batch), np.array(y_batch)
                        X_batch,y_batch = ([], [])
