from corpus_reader import  CorpusReader
import gensim
from nltk import word_tokenize
from nltk import sent_tokenize
import pandas as pd
from pprint import pprint

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

    def get_potential_nuggets(self, paragraph, max_len=8):
        nugget_candidates = []
        for sent in sent_tokenize(paragraph):
            words = word_tokenize(sent)
            #add all combinations
            for i0, word in enumerate(words):
                max_i = min(max_len + i0, len(words))
                for i1 in range(i0, max_i):
                    nugget_candidates.append(words[i0: i1])
        pprint(nugget_candidates[:20])
        # return pd.DataFrame(nugget_candidates, columns=['nugget_candidate'])
        return nugget_candidates

#TODO: 'I&#65533;&#65533;m???
# Use generator instead?
# Use hierarchical pandas index and preallocate

if __name__ == '__main__':
    r = CorpusReader()
    n = Nugget_Classifier(r)
    print(n.get_potential_nuggets(list(r.paragraphs.values())[0][0], max_len=3))


