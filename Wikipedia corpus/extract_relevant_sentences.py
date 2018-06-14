import os
import codecs
import re

base_directory = 'hMDS/english/'
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
        nugget_file = open(nugget_path, 'r')
        nuggets_raw = nugget_file.readlines()
        nuggets = []
        for n in nuggets_raw:
            id, nugget = re.split('^[0-9]+:', n, maxsplit=1)
            nugget = nugget.rstrip()
            nuggets.append(nugget)

        # access all texts and extract sentences which contain at least one nugget
        relevant_sentences = []
        for text_file_name in os.listdir(base_directory + file_name + '/input'):
            if (re.match('M[0-9]+.txt', text_file_name)):
                text_file = open(base_directory + file_name + '/input/' + text_file_name)
                text = [l.rstrip() for l in text_file.readlines()]
                for sent in text:
                    for nug in nuggets:
                        if (nug in sent):
                            relevant_sentences.append(sent)
                            break

        # create tuple: (query, nuggets, sentences)
        topic_tuples.append((query, nuggets, relevant_sentences))
