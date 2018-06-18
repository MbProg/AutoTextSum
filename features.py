import nltk
from nltk import FreqDist
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import nps_chat

background_corpus = [w.lower() for w in reuters.words() + brown.words() if w not in stopwords.words('english')]


# String t: term
# List(String) domain_texts: contains all texts for one query
# TODO: doesn't work!
def get_domain_relevance(term, paragraphs):
    fdist = FreqDist()
    nr_domains_containing_t = 0
    for text in paragraphs:
        tokens = nltk.word_tokenize(text)
        for t in tokens:
            fdist[t] +=1
    # (# of term t in domain d) / (# of terms in domain d)
    term_frequency = fdist[term]/fdist.N()
    # (# of domains containing t) / (# of domains)
    # 1 at the beginning because t is at least contained in its own domain
    nr_domains_containg_t = 1
    reuter_domains = reuters.categories()
    for domain in reuter_domains:
        if term in reuters.words(categories=[domain]):
            nr_domains_containg_t += 1
    inverse_domain_frequency = nr_domains_containing_t/len(reuter_domains)
    domain_relevance = term_frequency*inverse_domain_frequency
    return domain_relevance

# works
def get_term_frequency(term, paragraphs):
    fdist = FreqDist()
    for text in paragraphs:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        for t in tokens:
            fdist[t.lower()] +=1
    # (# of term t in domain d) / (# of terms in domain d)
    term_frequency = fdist[term]/fdist.N()
    return term_frequency*100

def get_common_frequency(word):
    fdist = FreqDist(background_corpus)
    return fdist[word]/fdist.N()*100000

# TODO: position features
# position of a paragraph is considered as position of the term
#def get_position():

#def is_at_beginning():

#def is_in_middle():

#def is_at_the_end():

#def get_tfidf_score():

#def get_pos_tag(term, sentence):

#def is_proper_name():

#def is_noun():

# use WordNet -> path_similarity
# return average similarity to query words
# works
def get_similarity_to_query(term, query):
    pairs = []
    similarities = []
    # TODO: use stemming?
    # remove stopwords and punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    for w in tokenizer.tokenize(query):
        if (w not in stopwords.words('english')):
            pairs.append((term, w))

    print('Pairs: ', pairs)
    sublist = []
    for pair in pairs:
        tmp = []
        for synset1 in wn.synsets(pair[0]):
            for synset2 in wn.synsets(pair[1]):
                #print(synset1)
                #print(synset2)
                if (synset1.path_similarity(synset2)) != None:
                    tmp.append(synset1.path_similarity(synset2))

        # wenn nicht mal synsets für ein Wort gefunden wurden, ist die Ähnlichkeit zum anderen Wort bestimmt sehr niedrig
        if (tmp == []):
            similarities.append(0)
        else:
            similarities.append(max(tmp))

    print(sum(similarities)/len(similarities))


# ------------------------------------------------------
# Platz zum Testen

paragraphs = ['Attention deficit hyperactivity disorder (ADHD) is a mental disorder of the neurodevelopmental type.[9][10] It is characterized by problems paying attention, excessive activity, or difficulty controlling behavior which is not appropriate for a person\'s age.',
                'The symptoms appear before a person is twelve years old, are present for more than six months, and cause problems in at least two settings (such as school, home, or recreational activities).',
                'In children, problems paying attention may result in poor school performance.',
                'Although it causes impairment, particularly in modern society, many children with ADHD have a good attention span for tasks they find interesting.']

print(get_term_frequency('attention', paragraphs))
print('Common frequncy(book)', get_common_frequency('book'))
print('Common frequncy(concentration)', get_common_frequency('concentration'))
print('Common frequncy(beech)', get_common_frequency('beech'))
print('Common frequncy(halibut)', get_common_frequency('halibut'))
