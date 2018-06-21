from nugget_classifier import Nugget_Classifier
import numpy as np
from nltk import word_tokenize
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import gensim
from corpus_reader import CorpusReader

word2vec =  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocab = Counter()
nb_words = 25000
nb_batches = 10000
max_sequence_length = 30
reader = CorpusReader()
words = set([])
for i in range(len(reader.topics)):
    text_id, topic = reader.topics.ix[i].text_id, reader.topics.ix[i].topic
    for paragraph, _ in reader.get_paragraph_nugget_pairs(str(text_id)):
        paragraph_words = word_tokenize(paragraph)
        vocab.update(paragraph_words)

word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(nb_words))}
print("built the index")
n = Nugget_Classifier(reader=reader)

we_matrix = (np.random.rand(nb_words, 300) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= nb_words:
        continue
    try:
        embedding_vector = word2vec[word]
        # words not found in embedding index will be all-zeros.
        we_matrix[i] = embedding_vector
    except:
        pass

print('built the word embedding matrix')
# producing validation dataset
x1_valid = []
x2_valid = []
y_valid = []
for i in range(209815):
    f = open('Data/queries/query_{}'.format(i))
    queries = eval(f.read())
    f.close()
    skip = True
    for q in queries:
        # skip if we have topics from the training data
        if q in ["parents concerns about religious classes at school", "parents deal with children's obesity"]:
            skip = False
    if skip:
        continue
    else:
        f = open('Data/nuggets/nuggets_{}'.format(i))
        nuggets = eval(f.read())
        y = np.load('Data/labels/y_{}.npy'.format(i))
        f.close()
        nuggets = [[word_index.get(word, len(word_index.keys())+1) for word in l] for l in nuggets]
        queries = [[word_index.get(word, len(word_index.keys())+1) for word in word_tokenize(l)] for l in queries]
        x1_valid += nuggets
        x2_valid += queries
        y_valid += list(y)
        if len(x1_valid)>=30000:
            break
x1_valid = pad_sequences(x1_valid, maxlen=30, padding='post')
x2_valid = pad_sequences(x2_valid, maxlen=30, padding='post')
y_valid = np.array(y_valid)
print(y_valid.shape)

n.build_network(hidden_dim=256, GRU_dim=256, nb_words=25000, max_sequence_length=30, we_matrix=we_matrix)
eval_batch = 0
best_score = 10000000000
for i in range(nb_batches):
    batch_id = np.random.randint(0, 209815)
    f = open('Data/queries/query_{}'.format(batch_id))
    queries = eval(f.read())
    f.close()
    skip = False
    for q in queries:
        # skip if we accidently got a batch of the dev set
        if q in ["parents concerns about religious classes at school", "parents deal with children's obesity"]:
            skip = True
    if skip:
        continue
    f = open('Data/nuggets/nuggets_{}'.format(batch_id))
    nuggets = eval(f.read())
    y = np.load('Data/labels/y_{}.npy'.format(batch_id))
    # converting the words to their indices in the vocabulary
    nuggets = [[word_index.get(word, len(word_index.keys())+1) for word in l] for l in nuggets]
    queries = [[word_index.get(word, len(word_index.keys())+1) for word in word_tokenize(l)] for l in queries]
    X = pad_sequences(nuggets, maxlen=30, padding='post')
    X2 = pad_sequences(queries, maxlen=30, padding='post')
    n.model.train_on_batch([X, X2], y)
    #print("batch {} processed".format(i))
    if i> eval_batch:
        _, mse = n.model.evaluate([x1_valid, x2_valid], y_valid)
        print("dev set score {}".format(mse))
        eval_batch += 500
        if mse<best_score:
            best_score = mse
            n.model.save_weights('network_weights')
