import os
import pandas as pd
import re
from collections import Counter
from functools import reduce
from nltk import word_tokenize
from nltk import sent_tokenize

class CorpusReader:
    # if not specified the reader assumes that it is placed directly in the unzipped corpus directory
    def __init__(self,
                 nugget_path=os.getcwd() +'/Corpus/AMTAllNuggets/',
                 paragraph_path=os.getcwd() + '/Corpus/SourceDocuments/',
                 topics_path=os.getcwd() +'/Corpus/SourceDocuments/' + 'Topics.txt'):
        '''Initializes the corpus reader. Saves the corresponding topics for each text, a dictionary of paragraphs with a list of
        paragraphs for each text_id and a nugget dictionary that for each text saves a dictionary of all workers and their chosen nuggets.
        '''
        self.build_paragraphs(paragraph_path)
        self.build_nuggets(nugget_path)
        self.topics = pd.read_csv(topics_path,sep='\t', names=['text_id', 'topic'])
        self.build_paragraph_word_scores()
        # Vocabulary takes too long right now and is not necessary yet for only word embedding features
        #self.build_vocabulary()

    def build_vocabulary(self):
        vocabulary = set([])
        for _,v in self.paragraphs.items():
            for paragraph_list in v:
                for paragraph in paragraph_list:
                    for word in word_tokenize(paragraph):
                        vocabulary = vocabulary.union(set([word.lower()]))
        self.vocabulary = vocabulary

    def build_paragraphs(self, paragraph_path):
        # filter out all files other than the paragraph files
        source_documents = [f for f in os.listdir(paragraph_path) if '1' in f]
        paragraphs = {}
        for file_name in source_documents:
            print(file_name)
            # document id is the last 4 characters, initialize a list for each id
            with open(paragraph_path + file_name,'r') as f:
                txt = f.read()
                paragraph_text = re.findall('<paragraph.*?</paragraph>',txt)
                # removing the paragraph tags
                paragraphs[file_name[-8:-4]] = [re.sub('<paragraph.*?>|</paragraph>','',paragraph) for paragraph in paragraph_text]
        self.paragraphs = paragraphs

    def build_nuggets(self, nugget_path):
        nugget_files = [f for f in os.listdir(nugget_path)]
        nuggets = {}
        for file_name in nugget_files:
            # read the selected nuggets into a pandas dataframe
            df = pd.read_csv(nugget_path + file_name,sep='\t', names=['worker_id', 'nugget'])
            # groupby workers and save all their nuggets in a list
            df2 = df.groupby('worker_id')['nugget'].apply(list)
            nuggets[file_name[:4]] = df2.to_dict()
        self.nuggets = nuggets

    def get_paragraph_nugget_pairs(self, text_id, tokenize_before_hash = False):
        ''' Returns a list of tuples of paragraphs and a dictionary with the corresponding nuggets as keys
        and the amount of workers that chose that nugget as value.

        Args:
            tokenize_before_hash: either have the resulting dict like this without tokenization before:
                {hash('hello world'): 123}
                or
                {hash(str(['hello', 'world']): 123}
            I needed the second one to get labels of nuggets AFTER tokenizing them
        '''
        paragraphs = self.paragraphs[text_id]
        # get a count of each nugget
        nuggets = Counter(reduce(lambda x,y: x+y, self.nuggets[text_id].values()))
        paragraph_nugget_tuples = []
        for paragraph in paragraphs:
            paragraph_nuggets = {}
            for nugget in nuggets.keys():
                try:
                    if tokenize_before_hash:
                        if nugget in paragraph:
                            paragraph_nuggets[repr(word_tokenize(nugget))] = nuggets[nugget]
                    else:
                        if nugget in paragraph:
                            paragraph_nuggets[nugget] = nuggets[nugget]
                except:
                    1
                    #print(nugget)
            paragraph_nugget_tuples.append((paragraph, paragraph_nuggets))
        return paragraph_nugget_tuples

    def build_paragraph_word_scores(self):
        '''Returns a list of all paragraphs where each word is assigned to a score
        that represents how many workers had chosen the word to be part of a nugget.
        '''
        # iterating through all topics
        data = []
        max_occurrence = 0
        for i in range(len(self.topics)):
            text_id, topic = self.topics.ix[i].text_id, self.topics.ix[i].topic
            paragraph_nugget_pairs = self.get_paragraph_nugget_pairs(str(text_id))
            for paragraph, nuggets in paragraph_nugget_pairs:
                paragraphs = []
                for sent in sent_tokenize(paragraph):
                    sentence_words = word_tokenize(paragraph)
                    sentence_word_occurrences = [[word,0] for word in sentence_words]
                    # iterating over all nuggets for that paragraph
                    for nugget in nuggets:
                        nugget_words = word_tokenize(nugget)
                        for j in range(len(sentence_words)):
                            # if the nugget is found in the paragraph increment all values of those nugget words by 1
                            if sentence_words[j:j+len(nugget_words)] == nugget_words:
                                for k in range(j, j+len(nugget_words)):
                                    sentence_word_occurrences[k][1] += 1
                                    if sentence_word_occurrences[k][1] > max_occurrence:
                                        max_occurrence = sentence_word_occurrences[k][1]
                                break
                    paragraphs.append(sentence_word_occurrences)
                data.append((i, paragraphs))
        self.max_occurrence = max_occurrence
        self.data = data
