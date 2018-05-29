import gensim
from nltk import word_tokenize
from nltk import sent_tokenize
from pprint import pprint
import numpy as np

class SimpleFeatureBuilder:

    def __init__(self, corpus_reader, train_size=0.8, batch_size=64, embedding_dim=300, limit_embeddings = None):
        self.word2vec =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', limit = limit_embeddings, binary=True)
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

    # Approach 2
    # TODO paragraphwise input == biased optimization? shuffle all?
    # TODO fix extreme class imbalance, maybe take 20% true nuggets and 80% brute force, currently more like 95:5%
    def __get_potential_nuggets__(self, paragraph, max_len=None, fixed_len = None):
        '''
        Helper for Task2. either max_len for all possible combinations of nuggets in the paragraph or fixed_len to only get those
        :param paragraph:
        :param max_len: int
        :param fixed_len: int
        :return: list(list(words))
        '''
        nugget_candidates = []
        # only one of either
        assert max_len is None or fixed_len is None
        for sent in sent_tokenize(paragraph):
            words = word_tokenize(sent)
            if fixed_len:
                for i0, word in enumerate(words):
                    if i0 + fixed_len < len(words):
                        nugget_candidates.append(words[i0 : i0 + fixed_len])
            #add all combinations
            if max_len:
                for i0, word in enumerate(words):
                    max_i = min(max_len + i0, len(words))
                    for i1 in range(i0, max_i):
                        nugget_candidates.append(words[i0: i1])
        # pprint(nugget_candidates[:5])
        # return pd.DataFrame(nugget_candidates, columns=['nugget_candidate'])
        return nugget_candidates

    def generate_sequence_word_embeddings(self, max_len=8, shuffle = False, seed = np.random.randint(1, 50)):
        '''
        Using Bucketing, i.e. having the same sequence length for each batch (curr_bucket) to make LSTM implementation
        easier.
        :param max_len: max length of potential nuggets
        :return: yield a batch of X shaped (batch_size, embedding_dim, curr_bucket)
            and y labels of the number of workers marking the nugget as relevant.
        '''
        # Initialize the features and labels
        X_batch,y_batch = ([], [])
        np.random.seed(seed)
        while True:
            for i in range(len(self.corpus_reader.topics)):
                text_id, topic = self.corpus_reader.topics.ix[i].text_id, self.corpus_reader.topics.ix[i].topic
                # TODO query as additional input?
                # topic_words = word_tokenize(self.corpus_reader.topics.ix[topic_index].topic)
                # topic_word_embeddings = np.average([self.get_word2vec(word) for word in topic_words],0)
                for paragraph, paragraph_nuggets in self.corpus_reader.get_paragraph_nugget_pairs(str(text_id), tokenize_before_hash= True ):
                    curr_bucket = np.random.randint(1, max_len)
                    nugget_candidates = self.__get_potential_nuggets__(paragraph, fixed_len = curr_bucket)
                    for candidate in nugget_candidates:
                        worker_count = 0
                        if repr(candidate) in paragraph_nuggets:
                            worker_count = paragraph_nuggets[repr(candidate)]
                        X_batch.append([self.get_word2vec(word) for word in candidate])
                        y_batch.append(worker_count)
                        if len(X_batch) == self.batch_size:
                            yield np.array(X_batch), np.array(y_batch)
                            X_batch, y_batch = ([], [])
                            break
