import gensim
from nltk import word_tokenize
from nltk import sent_tokenize
from pprint import pprint
import numpy as np
from nltk.tokenize.moses import MosesDetokenizer
detokenize = MosesDetokenizer().detokenize

import tensorflow as tf
import tensorflow_hub as hub

class SimpleFeatureBuilder:

    def __init__(self, corpus_reader, word_vectors_path='GoogleNews-vectors-negative300.bin',train_size=0.8, batch_size=64, embedding_dim=300, limit_embeddings = None):
        self.word2vec =  gensim.models.KeyedVectors.load_word2vec_format(word_vectors_path, limit = limit_embeddings, binary=True)
        self.corpus_reader = corpus_reader
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
        # Import the Universal Sentence Encoder's TF Hub module
        self.universal_sentence_encoder = hub.Module(module_url)
        #init later if needed
        self.embedding_session = None

    def get_word2vec(self, word):
        try:
            return self.word2vec[word]
        except:
            #if the word does not exist return a zero vector
            return np.zeros(self.embedding_dim)

    def get_word_embedding_features(self, mode='train', data_format='batch'):
        '''
        Returns training instances iteratively that can be build by iterating through the corpus. Each instance is the average word embedding of a given
        word in a sentence and its two surrounding words and the average of the topic/query words. The labels consist of the count of how many workers did choose to put that word in a
        nugget.
        '''
        temp_batch_size = self.batch_size
        if mode == 'train':
            paragraph_word_scores = self.corpus_reader.train_set
        else:
            self.batch_size = 1
            paragraph_word_scores = self.corpus_reader.dev_set
        # Initialize the features and labels
        X_batch,y_batch = ([], [])
        for topic_index, paragraph in paragraph_word_scores:
            topic_words = word_tokenize(self.corpus_reader.topics.ix[topic_index].topic)
            topic_word_embeddings = np.average([self.get_word2vec(word) for word in topic_words],0)
            for sentence_word_occurrences in paragraph:
                if len(sentence_word_occurrences)>1:
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
                        if data_format == 'batch' and len(X_batch) == self.batch_size:
                            if mode=='train':
                                yield np.array(X_batch), np.array(y_batch)
                            else:
                                end_sentence = False
                                if i == len(sentence_word_occurrences)-1:
                                    end_sentence = True
                                yield np.array(X_batch), np.array(y_batch), word, count, end_sentence
                            X_batch,y_batch = ([], [])
        # reset batch_size in case we were evaluating
        self.batch_size = temp_batch_size


    def get_complete_word_embedding_features(self, mode='train'):
        '''Returns the whole dataset in one array.
        '''
        if mode == 'train':
            paragraph_word_scores = self.corpus_reader.train_set
        else:
            paragraph_word_scores = self.corpus_reader.dev_set
        # Initialize the features and labels
        X_batch,y_batch = ([], [])
        for topic_index, paragraph in paragraph_word_scores:
            topic_words = word_tokenize(self.corpus_reader.topics.ix[topic_index].topic)
            topic_word_embeddings = np.average([self.get_word2vec(word) for word in topic_words],0)
            for sentence_word_occurrences in paragraph:
                #print(sentence_word_occurrences)
                if len(sentence_word_occurrences)>1:
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
        return np.array(X_batch), np.array(y_batch)

    # Approach 2
    # TODO paragraphwise input == biased optimization? shuffle all?
    # TODO fix extreme class imbalance, maybe take 20% true nuggets and 80% brute force, currently more like 95:5%
    # TODO sentence embeddigns (with yield)
    def __get_potential_nuggets__(self, paragraph, max_len=None, fixed_len = None):
        '''
        Helper for Task2. either max_len for all possible combinations of nuggets in the paragraph or fixed_len to only get those
        :param paragraph: str
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
                    for i1 in range(i0 + 1, max_i):
                        nugget_candidates.append(words[i0: i1])
        # pprint(nugget_candidates[:5])
        # return pd.DataFrame(nugget_candidates, columns=['nugget_candidate'])
        return nugget_candidates

    def generate_sentence_embeddings(self, sentences, tokenized = True, batch_size ='all'):
        '''
        generate Sentence Embeddings from a list of sentences using Google's Universal Sentence Encoder
            see https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1
        '''

        with tf.device("/cpu:0"):
            #if self.embedding_session is None:
            embedding_session = tf.Session()
            embedding_session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            if tokenized:
                sentences = [detokenize(sent, return_str=True) for sent in sentences]
            embeddings = np.array(embedding_session.run(self.universal_sentence_encoder(sentences)))
            return embeddings



    def generate_sequence_word_embeddings(self, max_len=8, min_class_percentage = 0.1, seed = np.random.randint(1, 50)):
        '''
        Using Bucketing, i.e. having the same sequence length for each batch (curr_bucket) to make LSTM implementation
        easier.
        TODO get all possible sequences? if approach actually works well
        :param max_len: max length of potential nuggets
            min_class_percentage: percentage of samples with a label > 0. (Oversampling) Use None or 0 for no sampling.
        :return: yield a batch of X shaped (batch_size, embedding_dim, curr_bucket)
            and y labels of the number of workers marking the nugget as relevant.
        '''
        # Initialize the features and labels
        X_batch,y_batch, word_sequence, queries, query_embeddings = ([], [], [], [],[])
        np.random.seed(seed)
        while True:
            for i in range(len(self.corpus_reader.topics)):
                text_id, topic = self.corpus_reader.topics.ix[i].text_id, self.corpus_reader.topics.ix[i].topic
                if topic not in self.corpus_reader.devset_topics:
                    for paragraph, paragraph_nuggets in self.corpus_reader.get_paragraph_nugget_pairs(str(text_id), tokenize_before_hash= True ):
                        #clean &#65533;
                        paragraph = paragraph.replace('&#65533;', '\'')
                        curr_bucket = np.random.randint(1, max_len)
                        nugget_candidates = self.__get_potential_nuggets__(paragraph, fixed_len = curr_bucket)
                        # fix class imbalance (fixed class percentage)... much more hacky than i thought it would be
                        if min_class_percentage:
                            num_true_nuggets = len(paragraph_nuggets)
                            max_random_nuggets = int(num_true_nuggets / min_class_percentage - num_true_nuggets)
                            np.random.shuffle(nugget_candidates)
                            nugget_candidates = nugget_candidates[:max_random_nuggets]
                            #get true nuggets
                            nugget_candidates_repr = [repr(candidate) for candidate in nugget_candidates]
                            true_nuggets = []
                            for true_nugget in list(paragraph_nuggets.keys()):
                                if true_nugget in nugget_candidates_repr:
                                    true_nugget = nugget_candidates[nugget_candidates_repr.index(true_nugget)]
                                    true_nuggets.append(true_nugget)
                            #balance again
                            max_random_nuggets = int(len(true_nuggets) / min_class_percentage - len(true_nuggets))
                            nugget_candidates = nugget_candidates[:max_random_nuggets]
                            #merge both
                            nugget_candidates += true_nuggets
                            np.random.shuffle(nugget_candidates)
                        for candidate in nugget_candidates:
                            worker_count = 0
                            if repr(candidate) in paragraph_nuggets:
                                worker_count = paragraph_nuggets[repr(candidate)]
                            word_sequence.append(candidate)
                            queries.append(topic)
                            query_embeddings.append(self.generate_sentence_embeddings([topic], tokenized=False))
                            X_batch.append([self.get_word2vec(word) for word in candidate])
                            y_batch.append(worker_count)
                            if len(X_batch) == self.batch_size:
                                yield np.array(X_batch), np.array(y_batch), word_sequence, queries, query_embeddings
                                X_batch, y_batch, word_sequence, queries, query_embeddings = [], [], [], [], []
                                break
