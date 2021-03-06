import os
import pandas as pd
import re
from collections import Counter
from functools import reduce
from nltk import word_tokenize
from nltk import sent_tokenize
import xml.etree.ElementTree as ET
import re
import logging
import io
import operator

class CorpusReader:
    # if not specified the reader assumes that it is placed directly in the unzipped corpus directory
    def __init__(self,
                 nugget_path=os.getcwd() +'/Corpus/AMTAllNuggets/',
                 paragraph_path=os.getcwd() + '/Corpus/SourceDocuments/',
                 topics_path=os.getcwd() +'/Corpus/SourceDocuments/' + 'Topics.txt',
                 approach='word'):
        '''Initializes the corpus reader. Saves the corresponding topics for each text, a dictionary of paragraphs with a list of
        paragraphs for each text_id and a nugget dictionary that for each text saves a dictionary of all workers and their chosen nuggets.
        '''
        if topics_path:
            self.topics = pd.read_csv(topics_path,sep='\t', names=['text_id', 'topic'])
            self.topics_path_exists = True
        else:
            self.topics_path_exists = False
        # self.endResultPrepare(paragraph_path);
        self.tagSentences(paragraph_path)
        self.build_paragraphs(paragraph_path)
        if topics_path:
            self.build_nuggets(nugget_path)
        self.devset_topics = [x for x in range(len(self.topics))][-2:]
        if approach=='word' and topics_path:
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
        queries = {}
        for file_name in source_documents:
            print(file_name)
            # document id is the last 4 characters, initialize a list for each id
            with open(paragraph_path + file_name,'r') as f:
                txt = f.read()
                paragraph_text = re.findall('<paragraph.*?</paragraph>',txt)
                # removing the paragraph tags
                paragraphs[file_name[-8:-4]] = [re.sub('<paragraph.*?>|</paragraph>','',paragraph) for paragraph in paragraph_text]
                if not self.topics_path_exists:
                    query = re.findall('<query>.*?</query>',txt)[0][7:-8]
                    queries[file_name[-8:-4]] = query
        self.paragraphs = paragraphs
        if not self.topics_path_exists:
            self.topics = pd.DataFrame.from_dict(queries, orient='index', columns=['topic']).rename(columns={0:'text_id'})
            self.topics['text_id'] = self.topics.index
            self.topics.reset_index(inplace=True, drop=True)

    def build_nuggets(self, nugget_path):
        nugget_files = [f for f in os.listdir(nugget_path)]
        nuggets = {}
        for file_name in nugget_files:
            # read the selected nuggets into a pandas dataframe
            df = pd.read_csv(nugget_path + file_name,sep='\t', names=['worker_id', 'nugget'])
            # groupby workers and save all their nuggets in a list

            # remove noise
            # remove records where nuggets are null
            # remove records which contain several sentences
            df.dropna(inplace=True)
            df['pointCount'] = df.nugget.apply(lambda x: x.count('.')+x.count('!')+x.count('?'))
            df = df[df['pointCount']<=1]

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
        text_id = str(text_id)
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
                            nugget = ' '.join([w for w in word_tokenize(nugget)])
                            paragraph_nuggets[nugget] = nuggets[nugget]
                except:
                    pass
                    #print(nugget)
            # make sure puncuation is surrounded by whitespace
            paragraph = re.sub('\.', ' . ', paragraph)
            paragraph = re.sub(',', ' , ', paragraph)
            paragraph_nugget_tuples.append((paragraph, paragraph_nuggets))
        return paragraph_nugget_tuples

    def build_paragraph_word_scores(self):
        '''Returns a list of all paragraphs where each word is assigned to a score
        that represents how many workers had chosen the word to be part of a nugget.
        '''
        # iterating through all topics
        train_set = []
        dev_set = []
        max_occurrence = 0
        total_words = 0
        for i in range(len(self.topics)):
            text_id, topic = self.topics.ix[i].text_id, self.topics.ix[i].topic
            paragraph_nugget_pairs = self.get_paragraph_nugget_pairs(str(text_id))
            for paragraph, nuggets in paragraph_nugget_pairs:
                paragraphs = []
                for sent in sent_tokenize(paragraph):
                    sentence_words = word_tokenize(sent)
                    # count how many words we have in total
                    if not i in self.devset_topics:
                        total_words += len(sentence_words)
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
                if i in self.devset_topics:
                    dev_set.append((i, paragraphs))
                else:
                    train_set.append((i, paragraphs))
        print(max_occurrence, len(train_set), len(dev_set))
        self.max_occurrence = max_occurrence
        self.train_set = train_set
        self.dev_set = dev_set
        self.total_words = total_words

    def tagSentences(self, document_path):
        '''
        fills self.sentences_dict which is dictionary
        key: (QueryID,DocumentIncrementerID started by 1, SentenceID started by 1)
        value: sentence in the content tag
        '''
        source_documents = [f for f in os.listdir(document_path) if '1' in f]
        # sentences:
        # key: (QueryID,DocumentIncrementerID started by 1, SentenceID started by 1)
        self.sentences_dict={}
        for file_name in source_documents:

            try:
                tree = ET.parse(document_path + file_name)
                QueryID = tree.getroot().attrib['queryID']
                t_documents = tree.getroot()[0]
                SentenceID = 0
                for DocumentID,document in enumerate(t_documents):
                    if document.tag == 'paragraph':
                        break
                    for sentences in document:
                        for sentence in sentences:
                            SentenceID+=1
                            self.sentences_dict[(QueryID,DocumentID+1,SentenceID)] = sentence.find('content').text



            except OSError as err:
                print("OS error: {0}".format(err))
                
    def endResultPrepare(self,document_path):
        source_documents = [f for f in os.listdir(document_path) if '1' in f]
        # self.SentenceCoveredByNugget is a dict
        # key = QueryID/DocumentID+1/SentenceID
        # value = (sentence, set of words in the nuggets)
        self.SentenceCoveredByNugget={}
        # 1. Fill self.SentenceCoveredByNugget by all the sentences in all the documents
        for file_name in source_documents:

            try:
                tree = ET.parse(document_path + file_name)
                QueryID = tree.getroot().attrib['queryID']
                t_documents = tree.getroot()[0]
                SentenceID = 0
                for DocumentID,document in enumerate(t_documents):
                    if document.tag == 'paragraph':
                        break
                    for sentences in document:
                        for sentence in sentences:
                            SentenceID+=1
                            self.SentenceCoveredByNugget['{0}/{1}/{2}'.format(QueryID,DocumentID+1,SentenceID)] = (sentence.find('content').text,set(()))
            except OSError as err:
                print("OS error: {0}".format(err))

            # 2. Now fill the second part of the tuple in the value in self.SentenceCoveredByNugget
            # by creating a set of all words for the specific sentence that have been in the nugget
            with open('nugget_predictions.txt','r') as f:
                for line in f:
                    linewords = re.split(r'\t+',line)
                    if linewords[0].rstrip() in self.SentenceCoveredByNugget:
                        self.SentenceCoveredByNugget[linewords[0].rstrip()] = (self.SentenceCoveredByNugget[linewords[0].rstrip()][0], \
                                                                                set(self.SentenceCoveredByNugget[linewords[0].rstrip()][1] | set(linewords[1].split())))
            # 3. Now we iterate thru self.SentenceCoveredByNugget to extract the sentences
            #    where the ratio of the words in the sentence detected in the nugget to the whole vocabulary size of the sentence is higher than the THRESHOLD
            # THRESHOLD = 0.4
            counter = 0
            sortedSentences={}
            for sentence_id in self.SentenceCoveredByNugget:
                sentenceVocabSize = len(set((self.SentenceCoveredByNugget[sentence_id][0].split())))
                nuggetVocabSize = len(self.SentenceCoveredByNugget[sentence_id][1])
                sortedSentences[sentence_id + ' \t ' + self.SentenceCoveredByNugget[sentence_id][0]] = nuggetVocabSize/sentenceVocabSize

            sortedSentencesDesc = sorted(sortedSentences.items(),key=operator.itemgetter(1),reverse=True)

            with io.open('nugget_predictions_final.txt', "a", encoding="utf-8") as f:
                for sentence,ratio in sortedSentencesDesc:
                    f.write(sentence + '\n')
                    counter += 1
                    if counter >= 30:
                        break
            print('Filename:',file_name," - Counter: ",counter)
            self.SentenceCoveredByNugget={}
