#!/usr/bin/env python3
from FeatureExtraction import *
from DataFunctions import *

def basic_tests(train_data, train_labels,       # Data for training classifier
                validate_data, validate_labels, # Test data & labels for ISOT
                FNN_data, FNN_labels,           # Test data & labels for FNN
                OriNews_data, OriNews_labels):  # Test data & labels for OriNews

    clf = train_random_foest(train_data, train_labels, 50)
    test_classifier(clf, validate_data, validate_labels, "RF: validate_data")
    test_classifier(clf, FNN_data, FNN_labels, "RF: FNN_data")
    test_classifier(clf, OriNews_data, OriNews_labels, "RF: OriNews_data")

    clf = train_NB(train_data, train_labels)
    test_classifier(clf, validate_data, validate_labels, "NB: validate_data")
    test_classifier(clf, FNN_data, FNN_labels, "NB: FNN_data")
    test_classifier(clf, OriNews_data, OriNews_labels, "NB: OriNews_data")

    clf = train_SVC(train_data, train_labels)
    test_classifier(clf, validate_data, validate_labels, "RF: validate_data")
    test_classifier(clf, FNN_data, FNN_labels, "RF: FNN_data")
    test_classifier(clf, OriNews_data, OriNews_labels, "RF: OriNews_data")


raw_data, labels = get_News_dataset()
FNN_raw_data, FNN_labels = get_FNN()
OriNews_raw_data, OriNews_labels = get_OriNews()

total_raw_data = raw_data+FNN_raw_data+OriNews_raw_data
raw_train_data, raw_validate_data, train_labels, validate_labels = split_data(raw_data, labels)


print("======================")
print("== Count_Ngram Only ==")
print("======================")

FNN_data, OriNews_data = get_CountVector_Ngram3(total_raw_data, FNN_raw_data, OriNews_raw_data)
train_data, validate_data = get_CountVector_Ngram3(total_raw_data, raw_train_data, raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("======================")
print("== Count_Word  Only ==")
print("======================")

FNN_data, OriNews_data = get_CountVector3(total_raw_data, FNN_raw_data, OriNews_raw_data)
train_data, validate_data = get_CountVector3(total_raw_data, raw_train_data, raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("======================")
print("==      ER Only     ==")
print("======================")

FNN_data = get_ER(FNN_raw_data)
OriNews_data = get_ER(OriNews_raw_data)
train_data = get_ER(raw_train_data)
validate_data = get_ER(raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews

print("======================")
print("==  Lemma + Count   ==")
print("======================")

FNN_data1 = tag_and_lem_list(FNN_raw_data)
OriNews_data1 = tag_and_lem_list(OriNews_raw_data)

raw_train_data, raw_validate_data, train_labels, validate_labels = split_data(raw_data, labels)
train_data1 = tag_and_lem_list(raw_train_data)
validate_data1 = tag_and_lem_list(raw_validate_data)

lemma_total_train = validate_data1+train_data1+FNN_data1+OriNews_data1

train_data, validate_data = get_CountVector3(lemma_total_train, train_data1, validate_data1)
FNN_data, OriNews_data = get_CountVector3(lemma_total_train, FNN_data1, OriNews_data1)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews



print("==========================")
print("== NLTK removed + Count ==")
print("==========================")

raw_data_stop = remove_NLTK_stop1(raw_data)
FNN_raw_data_stop = remove_NLTK_stop1(FNN_raw_data)
OriNews_raw_data_stop = remove_NLTK_stop1(OriNews_raw_data)

raw_train_data_stop, raw_validate_data_stop, train_labels, validate_labels = split_data(raw_data_stop, labels)


total_stop_data = raw_data_stop+FNN_raw_data_stop+OriNews_raw_data_stop

FNN_data, OriNews_data = get_CountVector_Ngram3(total_stop_data, FNN_raw_data_stop, OriNews_raw_data_stop)
train_data, validate_data = get_CountVector_Ngram3(total_stop_data, raw_train_data_stop, raw_validate_data_stop)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("===========================")
print("== spaCy removed + Count ==")
print("===========================")

raw_data_stop = remove_spaCy_stop1(raw_data)
FNN_raw_data_stop = remove_spaCy_stop1(FNN_raw_data)
OriNews_raw_data_stop = remove_spaCy_stop1(OriNews_raw_data)

raw_train_data_stop, raw_validate_data_stop, train_labels, validate_labels = split_data(raw_data_stop, labels)

FNN_data, OriNews_data = get_CountVector_Ngram3(total_stop_data, FNN_raw_data_stop, OriNews_raw_data_stop)
train_data, validate_data = get_CountVector_Ngram3(total_stop_data, raw_train_data_stop, raw_validate_data_stop)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("===========================")
print("==        PoS Only       ==")
print("===========================")

FNN_data = get_PoS(FNN_raw_data)
OriNews_data = get_PoS(OriNews_raw_data)
train_data = get_PoS(raw_train_data)
validate_data = get_PoS(raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("===========================")
print("==    TFIDF_Word Only    ==")
print("===========================")

FNN_data, OriNews_data = get_TFIDF_Word3(total_raw_data, FNN_raw_data, OriNews_raw_data)
train_data, validate_data = get_TFIDF_Word3(total_raw_data, raw_train_data, raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("===========================")
print("==    TFIDF_Ngram Only   ==")
print("===========================")

FNN_data, OriNews_data = get_TFIDF_NGram3(total_raw_data, FNN_raw_data, OriNews_raw_data)
train_data, validate_data = get_TFIDF_NGram3(total_raw_data, raw_train_data, raw_validate_data)

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews


print("===========================")
print("==       VADER Only      ==")
print("===========================")

FNN_data = make_VADER_score_non_neg(get_VADER_score(FNN_raw_data))
OriNews_data = make_VADER_score_non_neg(get_VADER_score(OriNews_raw_data))
train_data = make_VADER_score_non_neg(get_VADER_score(raw_train_data))
validate_data = make_VADER_score_non_neg(get_VADER_score(raw_validate_data))

basic_tests(train_data, train_labels,       # Data for training classifier
            validate_data, validate_labels, # Test data & labels for ISOT
            FNN_data, FNN_labels,           # Test data & labels for FNN
            OriNews_data, OriNews_labels)   # Test data & labels for OriNews
