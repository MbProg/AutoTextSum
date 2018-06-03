from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector
import matplotlib.pyplot as plt
from nltk import word_tokenize
import numpy as np
reader = CorpusReader(approach='word')
#Check if the reader correctly read the input
sentence_nugget_count = 0
nugget_lengths = []
nugget_count = 0
error_count = 0
for key in reader.nuggets.keys():
    for worker, nuggets in reader.nuggets[key].items():
        # a nugget should be a sentence if it starts uppercase and ends with ./?/!
        for nugget in nuggets:
            try:
                if nugget[0].isupper() and nugget[-1] in ['.', '!', '?']:
                    sentence_nugget_count += 1
                # tokenize to get word count
                nugget = word_tokenize(nugget)
                nugget_lengths.append(len(nugget))
                nugget_count += 1
            except:
                error_count +=1
plt.hist(nugget_lengths, bins= np.unique(nugget_lengths))
plt.show()
print("sentences make up {} of all nuggets".format(sentence_nugget_count/nugget_count))
print("{} have resulted in an error".format(error_count))
