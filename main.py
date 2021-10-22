# 1.0
import random
random.seed(123)
# 1.1
import codecs
# 1.5
import string
# 1.6
from nltk.stem.porter import PorterStemmer
import gensim
import os

stemmer = PorterStemmer()


def define_stop_words(stopwordfile):
    with codecs.open("stop_words.txt", "r", "utf-8") as f:
        stop_words_list = f.read().split(",")
        return stop_words_list


# 1. Data loading and preprocessing
def load_paragraphs(file):
    paragraphs = [text for text in file.read().split(2 * os.linesep) if (text != "") and ("gutenberg" not in text.lower())]
    #print(f"Paragraphs: {paragraphs[:50]}")
    return paragraphs


# 1.4 tokenize paragraphs
def tokenize_paragraphs(paragraphs):
    translator = str.maketrans('', '', string.punctuation + "\n\r\t")
    tokenized_paragraphs = [
        [word.lower().translate(translator) for word in paragraph.split(" ")] for paragraph in paragraphs
    ]
    #print(f"Tokenized paragraphs {tokenized_paragraphs[:50]}")
    return tokenized_paragraphs


# 1.6  Using portstemmer stem words
def stem_words(tokenized_paragraphs):
    stemmed_words = [[stemmer.stem(word) for word in paragraph] for paragraph in tokenized_paragraphs]
    #print(f"Stemmed words: {stemmed_words[:50]}")
    return stemmed_words


def create_corpus(stop_words, words):
    dictionary = gensim.corpora.Dictionary(words)
    stop_word_ids = [dictionary.token2id[stop_word] for stop_word in stop_words if stop_word in dictionary.values()]
    dictionary.filter_tokens(stop_word_ids)
    list_of_bow = [dictionary.doc2bow(paragraph) for paragraph in words]
    return list_of_bow, dictionary


def tf_idf_model(corpus, calulate_sim_matr=True):
    tfidf_model = gensim.models.TfidfModel(corpus)

    tfidf_corpus = tfidf_model[corpus]

    if calulate_sim_matr:
        tfidf_sim = gensim.similarities.MatrixSimilarity(tfidf_corpus)
        return tfidf_sim
    else:
        return tfidf_corpus


def tf_idf_model_LSI(corpus, dictionary):
    lsi_model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=100)

    lsi_corpus = lsi_model[corpus]

    lsi_sim = gensim.similarities.MatrixSimilarity(lsi_corpus)

    topics = lsi_model.show_topics()
    for topic in topics:
        print(topic)
    return lsi_sim


def prepare_query(queries):
    #translator = str.maketrans('', '', string.punctuation + "\n\r\t")
    #tokenized_query = [[word.lower().translate(translator) for word in query.split()] for query in queries]
    #stemmed_query = [stemmer.stem(word) for word in tokenized_query]
    tokenized_queries = tokenize_paragraphs(queries)
    stemmed_queries = stem_words(tokenized_queries)
    return stemmed_queries


def query_to_bow(prepared_queries, stop_words):
    list_of_bow, dictionary = create_corpus(stop_words, prepared_queries)
    return list_of_bow, dictionary


def task_1():
    with codecs.open("pg3300.txt", "r", "utf-8") as f:
        paragraphs = load_paragraphs(f)
        tokenized_paragraps = tokenize_paragraphs(paragraphs)
        stemmed_word_paragraphs = stem_words(tokenized_paragraps)
        return stemmed_word_paragraphs


#GLOBAL
stop_words = define_stop_words("stop_words.txt")


def task_2():
    words = task_1()
    corpus, dictionary = create_corpus(stop_words, words)
    return corpus, dictionary


def task_3():
    corpus, dictionary = task_2()

    tfidif_s_m = tf_idf_model(corpus)

    lsi_s_m = tf_idf_model_LSI(corpus, dictionary)

    print(tfidif_s_m)


def task_4():
    # to be able to reuse code, we send in a list of queries even when there is only one
    queries = ["What is the function of money?"]
    prepared_query = prepare_query(queries)
    print(prepared_query)
    list_of_bow, dictionary = query_to_bow(prepared_query, stop_words)
    print(list_of_bow)
    print(dictionary)
    tf_idf = tf_idf_model(list_of_bow, calulate_sim_matr=False)
    print([i for i in tf_idf])



if __name__ == '__main__':

    print("----------- Starting -----------")
    print("Task 1")
    #task_1()
    print("----------- Finished -----------")

    print()

    print("----------- Starting -----------")
    print("Task 2")
    #task_2()
    print("----------- Finished -----------")

    print()

    print("----------- Starting -----------")
    print("Task 3")
    #task_3()
    print("----------- Finished -----------")

    print()

    print("----------- Starting -----------")
    print("Task 4")
    task_4()
    print("----------- Finished -----------")

