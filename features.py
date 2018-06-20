class Features:

    import nltk
    from nltk import FreqDist



    #global variables
    domain_fdist = FreqDist()
    len_paragraphs = 0
    len_domain_fdist = 0
    background_corpus = []
    fdist_background_corpus = FreqDist
    stopwords_english = []
    paragraphs = []
    # tokenisierte Query
    query = []


    # String t: term
    # List(String) domain_texts: contains all texts for one query
    # TODO: doesn't work!
    def get_domain_relevance(term, paragraphs):
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
    # computes the term frequncy in one domain
    def get_term_frequency(self, term):
        # (# of term t in domain d) / (# of terms in domain d)
        term_frequency = self.domain_fdist[term]/self.domain_fdist.N()
        return term_frequency*100

    def get_common_frequency(self, word):
        return self.fdist_background_corpus[word]/self.fdist_background_corpus.N()*100000

    # TODO: position features
    # position of a paragraph is considered as position of the term
    def get_position(self, paragraph):
        return paragraphs.index(paragraph)

    def is_at_beginning(self, paragraph):
        pos = paragraphs.index(paragraph)
        text_length = len(paragraphs)
        if pos <= text_length/3:
            return 1
        else:
            return 0

    #def is_in_middle(paragraph):

    #def is_at_the_end(paragraph):

    #def get_tfidf_score():

    #def get_pos_tag(term, sentence):

    #def is_proper_name():

    #def is_noun():

    # use WordNet -> path_similarity
    # return average similarity to query words
    # works
    def get_similarity_to_query(self, term):
        pairs = []
        similarities = []
        for w in self.query:
            if (w not in self.stopwords_english):
                pairs.append((term, w))
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
        return sum(similarities)/len(similarities)



    def __init__(self, paragraphs, query):
        from nltk.corpus import brown
        from nltk.corpus import reuters
        from nltk.corpus import nps_chat
        from nltk.corpus import stopwords
        from nltk import word_tokenize
        from nltk.tokenize import RegexpTokenizer
        from nltk import FreqDist

        self.background_corpus = [w.lower() for w in reuters.words()[:1000] + brown.words()[:1000] + nps_chat.words()[:1000] if w not in stopwords.words('english')]
        self.fdist_background_corpus = FreqDist(self.background_corpus)
        #print(self.background_corpus)
        for text in paragraphs:
            tokens = word_tokenize(text)
            for t in tokens:
                self.domain_fdist[t] +=1
        self.stopwords_english = stopwords.words('english')
        global wn
        from nltk.corpus import wordnet as wn
        self.len_domain_fdist = self.domain_fdist.N()
        self.len_paragraphs = len(paragraphs)
        self.paragraphs = paragraphs
        tokenizer = RegexpTokenizer(r'\w+')
        self.query = tokenizer.tokenize(query)
        #print("Query: ", self.query)




    # ------------------------------------------------------
    # Platz zum Testen

paragraphs = ['Attention deficit hyperactivity disorder (ADHD) is a mental disorder of the neurodevelopmental type.[9][10] It is characterized by problems paying attention, excessive activity, or difficulty controlling behavior which is not appropriate for a person\'s age.',
                'The symptoms appear before a person is twelve years old, are present for more than six months, and cause problems in at least two settings (such as school, home, or recreational activities).',
                'In children, problems paying attention may result in poor school performance.',
                'Although it causes impairment, particularly in modern society, many children with ADHD have a good attention span for tasks they find interesting.']


f = Features(paragraphs, "action figure")

#print(f.get_term_frequency('attention'))
#print('Common frequncy(action)', f.get_common_frequency('wanna'))
#print('Common frequncy(book)', f.get_common_frequency('action'))
#print(f.get_similarity_to_query("action"))
#print('Common frequncy(beech)', get_common_frequency('beech'))
#print('Common frequncy(halibut)', get_common_frequency('halibut'))
print(f.is_at_beginning('Although it causes impairment, particularly in modern society, many children with ADHD have a good attention span for tasks they find interesting.'))
