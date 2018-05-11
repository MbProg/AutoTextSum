from corpus_reader import CorpusReader
from simple_feature_builder import SimpleFeatureBuilder
from simple_nugget_detector import SimpleNuggetDetector
reader = CorpusReader()
fb = SimpleFeatureBuilder(reader)
#Check if the reader correctly read the input
print(reader.nuggets.keys())
print(reader.paragraphs.keys())
print(reader.topics.ix[0].topic)
#initialize the model and train it
nugget_detector = SimpleNuggetDetector(reader, fb, model='lr', params=None)
nugget_detector.fit_model()
