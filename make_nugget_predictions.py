from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector
import os
from simple_evaluator import Evaluate
import time
from nltk import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
detokenize = MosesDetokenizer().detokenize

train_corpus_reader = CorpusReader(approach='word')
full_corpus_reader = CorpusReader(nugget_path=None,
                      paragraph_path=os.getcwd() + '/AutoTS_Corpus/',
                      topics_path=None,
                      approach='word')
fb = SimpleFeatureBuilder(train_corpus_reader, word_vectors_path='GoogleNews-vectors-negative300.bin')

params = {}
#params['n_estimators'] = 15
params['max_depth'] = 3
params['min_samples_split'] = 2
params['min_samples_leaf'] = 3
params['min_weight_fraction_leaf'] = 0

nugget_detector = SimpleNuggetDetector(train_corpus_reader, fb, model='tree', params=params)
nugget_detector.fit_model()

predictions = []
sentences_dict = OrderedDict(sorted(full_corpus_reader.sentences_dict.items(), key=lambda t: t[0][0]))
f = open('nugget_predictions.txt', 'w')
for (query_id,document_id,sentence_id),sent in sentences_dict.items():
    sentence_words = word_tokenize(sent)
    query = full_corpus_reader.topics[full_corpus_reader.topics['text_id']==str(query_id)].topic.values[0]
    query_embeddings = np.average([fb.get_word2vec(word) for word in word_tokenize(query)],0)
    sentence = []
    if len(sentence_words)>=3:
        for i,word in enumerate(sentence_words):
            if i > 0 and i < len(sentence_words)-1:
                surrounding_words = [sentence_words[i-1]] + [word] + [sentence_words[i+1]]
            elif i==0:
                surrounding_words = [word] + [sentence_words[i+1]]
            else:
                surrounding_words = [sentence_words[i-1]] + [word]
            surrounding_word_embeddings = np.average([fb.get_word2vec(word) for word in surrounding_words],0)
            X_batch = np.average([query_embeddings, surrounding_word_embeddings],0)
            prediction = nugget_detector.model.predict_proba(X_batch[None,:])
            prediction = nugget_detector.convert_predictions(prediction)
            predictions.append(prediction)
            sentence.append((word, prediction))
        nugget_predictions = nugget_detector.convert_word_to_nugget_predictions([sentence])
        for prediction in nugget_predictions:
            try:
                if len(prediction)>1:
                    f.write('{}/{}/{} \t '.format(query_id,document_id,sentence_id) + ' '.join(prediction) + '\n')
            except:
                pass
f.close()
