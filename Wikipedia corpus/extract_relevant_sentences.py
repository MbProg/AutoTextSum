import os
import codecs
import re
from nltk import word_tokenize
import numpy as np
import gensim
import sys
def get_wiki_corpus():
    base_directory = os.getcwd()+'/Wikipedia corpus/hMDS/english/'
    dict_id_to_query = {}
    topic_file = codecs.open(base_directory + 'topics.txt', 'r', 'iso-8859-1')
    topic_file_lines = [l.rstrip() for l in topic_file.readlines()]

    # can be used as an input for training summarization systems
    topic_tuples = []

    for l in (topic_file_lines):
        id, name = re.split("_", l, maxsplit=1)
        name = name.replace("_", " ")
        dict_id_to_query[id] = name

    for file_name in os.listdir(base_directory):
        if(file_name.startswith('D')):
            # store query
            query = dict_id_to_query[file_name]
            # store nuggets
            nugget_path = base_directory+file_name+'/metadata/nuggets.txt'
            nugget_file = open(nugget_path, 'r',encoding='utf-8')
            nuggets_raw = nugget_file.readlines()
            nuggets = []
            for n in nuggets_raw:
                id, nugget = re.split('^[0-9]+:', n, maxsplit=1)
                nugget = nugget.rstrip()
                nuggets.append(nugget)
            nugget_file.close()
            # access all texts and extract sentences which contain at least one nugget
            relevant_sentences = []
            for text_file_name in os.listdir(base_directory + file_name + '/input'):
                if (re.match('M[0-9]+.txt', text_file_name)):
                    text_file = open(base_directory + file_name + '/input/' + text_file_name, encoding='utf-8')
                    text = [l.rstrip() for l in text_file.readlines()]
                    text_file.close()
                    for sent in text:
                        for nug in nuggets:
                            if (nug in sent):
                                relevant_sentences.append(sent)
                                break

            # create tuple: (query, nuggets, sentences)
            topic_tuples.append((query, nuggets, relevant_sentences))

    return topic_tuples

def get_word2vec(word_2_vec, word):
    try:
        return word_2_vec[word]
    except:
        #if the word does not exist return a zero vector
        return np.zeros(300)

def wiki_corpus_to_word_features(label=1):
    topic_tuples = get_wiki_corpus()
    X, y = ([], [])
    word_2_vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    for topic, nuggets, sentences in topic_tuples:
        topic_word_embeddings = np.average([get_word2vec(word_2_vec, word) for word in word_tokenize(topic)],0)
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words)>2:
                current_nugget = ''
                for nugget in nuggets:
                    nugget_words = word_tokenize(nugget)
                    for i in range(len(sentence_words)):
                        if sentence_words[i:i+len(nugget_words)] == nugget_words:
                            current_nugget = nugget
                            nugget_indices = [j for j in range(i, i+len(nugget_words))]
                            #print(current_nugget)
                            # indices of nugget were found so we can skip the rest
                            break
                    # nugget was found so we can break
                    if current_nugget != '':
                        break
                for i,word in enumerate(sentence_words):
                    surrounding_words = []
                    if i > 0 and i < len(sentence_words)-1:
                        surrounding_words = [sentence_words[i-1]] + [word] + [sentence_words[i+1]]
                    elif i==0:
                        surrounding_words = [word] + [sentence_words[i+1]]
                    else:
                        surrounding_words = [sentence_words[i-1]] + [word]
                    word_label = label if i in nugget_indices else 0
                    surrounding_word_embeddings = np.average([get_word2vec(word_2_vec, word) for word in surrounding_words],0)
                    X.append(np.average([topic_word_embeddings, surrounding_word_embeddings],0))
                    y.append(word_label)
    return np.array(X), np.array(y)

print(wiki_corpus_to_word_features())
