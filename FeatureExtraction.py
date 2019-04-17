import spacy
from collections import Counter
from nltk import pos_tag
from nltk.data import load
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from TagLemmatize import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import en_core_web_sm
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


# count-word feature extaction
def get_CountVector3(all_data, train_data, test_data):
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(all_data)
    x_train_data =  count_vect.transform(train_data)
    x_test_data =  count_vect.transform(test_data)
    return x_train_data, x_test_data

def get_CountVector1(all_data):
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(all_data)
    return count_vect.transform(all_data)

def remove_NLTK_stop3(all_data, train_data, test_data):
    sw = stopwords.words('english')
    deto = Detok()

    all_cleaned = list()
    train_cleaned = list()
    test_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in train_data:
        word_tokens = word_tokenize(article)
        train_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in test_data:
        word_tokens = word_tokenize(article)
        test_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned, train_cleaned, test_cleaned


def remove_spaCy_stop3(all_data, train_data, test_data):
    spacy_nlp = spacy.load('en')
    sw = spacy.lang.en.stop_words.STOP_WORDS
    deto = Detok()

    all_cleaned = list()
    train_cleaned = list()
    test_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in train_data:
        word_tokens = word_tokenize(article)
        train_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    for article in test_data:
        word_tokens = word_tokenize(article)
        test_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned, train_cleaned, test_cleaned

def remove_spaCy_stop1(all_data):
    spacy_nlp = spacy.load('en')
    sw = spacy.lang.en.stop_words.STOP_WORDS
    deto = Detok()

    all_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned

def remove_NLTK_stop1(all_data):
    sw = stopwords.words('english')
    deto = Detok()

    all_cleaned = list()

    for article in all_data:
        word_tokens = word_tokenize(article) 
        all_cleaned.append(deto.detokenize(
            [w for w in word_tokens if not w in sw]))

    return all_cleaned


def get_CountVector_NLTK_Stop3(all_data, train_data, test_data):    
    sw = stopwords.words('english')
    count_vect = CountVectorizer(stop_words=sw)
    count_vect = count_vect.fit(all_data)
    x_train_data =  count_vect.transform(train_data)
    x_test_data =  count_vect.transform(test_data)
    return x_train_data, x_test_data

def get_CountVector_spaCy_Stop3(all_data, train_data, test_data):
    spacy_nlp = spacy.load('en')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    count_vect = CountVectorizer(stop_words=spacy_stopwords)
    count_vect = count_vect.fit(all_data)
    x_train_data =  count_vect.transform(train_data)
    x_test_data =  count_vect.transform(test_data)
    return x_train_data, x_test_data


# count-ngram feature extaction
def get_CountVector_Ngram3(all_data, train_data, test_data):
    count_vect = CountVectorizer(ngram_range=(2,3))
    count_vect = count_vect.fit(all_data)
    x_train_data =  count_vect.transform(train_data)
    x_test_data =  count_vect.transform(test_data)
    return x_train_data, x_test_data


def get_CountVector_Ngram1(all_data):
    count_vect = CountVectorizer(ngram_range=(2,3))
    count_vect = count_vect.fit(all_data)
    return count_vect.transform(all_data)
  

# TFIDF-word feature extraction
def get_TFIDF_Word3(all_data, train_data, test_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 max_features=5000)
    tfidf_vect.fit(all_data)
    x_train_data =  tfidf_vect.transform(train_data)
    x_test_data =  tfidf_vect.transform(test_data)
    return x_train_data, x_test_data

def get_TFIDF_Word1(all_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 max_features=5000)
    tfidf_vect.fit(all_data)
    return tfidf_vect.transform(all_data)


# TFIDF-ngram feature extraction
def get_TFIDF_NGram3(all_data, train_data, test_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 ngram_range=(2,3), max_features=5000)
    tfidf_vect.fit(all_data)
    x_train_data =  tfidf_vect.transform(train_data)
    x_test_data =  tfidf_vect.transform(test_data)
    return x_train_data, x_test_data


# TFIDF-ngram feature extraction
def get_TFIDF_NGram1(all_data):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 ngram_range=(2,3), max_features=5000)
    tfidf_vect.fit(all_data)
    return tfidf_vect.transform(all_data)


# VADER feature extraction
def get_VADER_score(data_list):
    analyser = SentimentIntensityAnalyzer()
    ret_list = list()
    for data in data_list:
        ret_list.append(list(analyser.polarity_scores(data).values()))
    return ret_list

def make_VADER_score_non_neg(article_list):
    ret_list = list()
    for article_vals in article_list:
        ret_list.append([x+1 for x in article_vals])
    return ret_list

def tag_and_lem_list(data_list):
    ret_list = []
    for d in data_list:
        ret_list.append(tag_and_lem(d))
    return ret_list

def get_PoS(all_data):
    # Turn all_data into PoS
    all_pos = list()
    for article in all_data:
        all_pos.append(pos_tag(word_tokenize(article)))

    # Create a counter for all_pos
    all_pos_counter = list()
    for article in all_pos:
        all_pos_counter.append(Counter( tag for word,  tag in article))

    all_pos_count = list()

    tagdict = load('help/tagsets/upenn_tagset.pickle')
    # Count up each PoS and giving a value of 0 to those that do not occur
    for counter in all_pos_counter:
        temp = list()
        for key in tagdict:
            temp.append(counter[key])
        all_pos_count.append(temp)

    return all_pos_count

def get_ER(all_data):
    named_entity_list = ("PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", 
                         "PRODUCT", "EVENT","WORK_OF_ART", "LAW", "LANGUAGE", 
                         "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", 
                         "ORDINAL", "CARDINAL")
    nlp = en_core_web_sm.load()

    all_list = list()

    # get entites
    for article in all_data:
        nlpa = nlp(article)
        all_list.append(Counter([(X.label_) for X in nlpa.ents]))     

    all_list_counts = list()

    for counter in all_list:
        temp = list()
        for entity in named_entity_list:
            temp.append(counter[entity])
        all_list_counts.append(temp)

    return all_list_counts