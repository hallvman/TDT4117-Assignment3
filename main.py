import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
import gensim
import os

random.seed(123)
stemmer = PorterStemmer()


def define_stop_words(stopwordfile):
    with codecs.open("stop_words.txt", "r", "utf-8") as f:
        stop_words_list = f.read().split(",")
        return stop_words_list


# 1. Data loading and preprocessing
def load_paragraphs(file):
    paragraphs = [text for text in file.read().split(2 * os.linesep)
                  if (text != "") and ("gutenberg" not in text.lower())]
    # print(f"Paragraphs: {paragraphs[:50]}")
    return paragraphs


# 1.4 tokenize paragraphs
def tokenize_paragraphs(paragraphs):
    translator = str.maketrans('', '', string.punctuation + "\n\r\t")
    tokenized_paragraphs = [
        [word.lower().translate(translator) for word in paragraph.split(" ")] for paragraph in paragraphs
    ]
    # print(f"Tokenized paragraphs {tokenized_paragraphs[:50]}")
    return tokenized_paragraphs


# 1.6  Using portstemmer stem words
def stem_words(tokenized_paragraphs):
    stemmed_words = [[stemmer.stem(word) for word in paragraph] for paragraph in tokenized_paragraphs]
    # print(f"Stemmed words: {stemmed_words[:50]}")
    return stemmed_words


# 2.1 dictionary building
# 2.1 Filter out stopwords using global value "stop_words"
# 2.2 map paragraphs into BOW
def create_corpus(stop_words, words):
    dictionary = gensim.corpora.Dictionary(words)
    stop_word_ids = [dictionary.token2id[stop_word] for stop_word in stop_words if stop_word in dictionary.values()]
    dictionary.filter_tokens(stop_word_ids)
    list_of_bow = [dictionary.doc2bow(paragraph) for paragraph in words]
    return list_of_bow, dictionary


# 3.1 Build TFIDF-model for documents
# 3.2 Map BOWs into tf-idf weights
# 3.3 Construct matrix similarity
def tf_idf_model(corpus, calulate_sim_matr=True):
    tfidf_model = gensim.models.TfidfModel(corpus)

    tfidf_corpus = tfidf_model[corpus]

    if calulate_sim_matr:
        tfidf_sim = gensim.similarities.MatrixSimilarity(tfidf_corpus)
        return tfidf_sim
    else:
        return tfidf_corpus


# Repeat 3.1, 3.2 and 3.3 for LSI-model
# Report top 3 topics
def tf_idf_model_LSI(corpus, dictionary):
    lsi_model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=100)

    lsi_corpus = lsi_model[corpus]

    lsi_sim = gensim.similarities.MatrixSimilarity(lsi_corpus)

    topics = lsi_model.show_topics()
    for topic in topics[:3]:
        print(topic)
    return lsi_sim

# 4.1
# Tokenize and stem query
def prepare_query(query):
    translator = str.maketrans('', '', string.punctuation + "\n\r\t")
    tokenized_query = [word.lower().translate(translator) for word in query.split()]
    stemmed_query = [stemmer.stem(word) for word in tokenized_query]
    return stemmed_query

# 4.2
# Convert query to BOW-representation
def query_to_bow(query, dictionary):
    prepared_query = prepare_query(query)
    query_bow = dictionary.doc2bow(prepared_query)
    return query_bow


def task_1():
    with codecs.open("pg3300.txt", "r", "utf-8") as f:
        paragraphs = load_paragraphs(f)
        tokenized_paragraps = tokenize_paragraphs(paragraphs)
        stemmed_word_paragraphs = stem_words(tokenized_paragraps)
        return stemmed_word_paragraphs, paragraphs


# GLOBAL
stop_words = define_stop_words("stop_words.txt")


def task_2():
    words, paragraphs = task_1()
    corpus, dictionary = create_corpus(stop_words, words)
    return corpus, dictionary, paragraphs


def task_3():
    corpus, dictionary, paragraphs = task_2()

    tfidif_s_m = tf_idf_model(corpus)

    lsi_s_m = tf_idf_model_LSI(corpus, dictionary)

    print(tfidif_s_m)

    print(lsi_s_m)


def task_4():

    #TF-IDF
    print(f"------------------TFIDF-------------------")

    # create corpus and dict from documents
    corpus, dictionary, paragraphs = task_2()

    # Create model (difficult reusing function from task 3
    tfidf_model = gensim.models.TfidfModel(corpus, dictionary=dictionary)

    # Prepare query
    query_bow = query_to_bow("How taxes influence Economics?", dictionary)

    # query-BOW => tf-idf weights
    tfidf_query = tfidf_model[query_bow]

    # Doc-bow => tf-idf weights
    tf_idf_corpus = tfidf_model[corpus]

    # Create sim matrix
    tfidf_index = gensim.similarities.MatrixSimilarity(tf_idf_corpus)

    doc2similarity_tfidf = enumerate(tfidf_index[tfidf_query])

    # Print 5 lines from top 3 docs
    doc2similarity_tfidf_sorted = sorted(doc2similarity_tfidf, key=lambda kv: -kv[1])
    for parnum, sim in doc2similarity_tfidf_sorted[:3]:
        current_paragraph = paragraphs[parnum].split("\n")[:5] if len(paragraphs[parnum].split("\n")) > 5 else paragraphs[parnum].split("\n")
        print(f"\n[paragraph: {parnum}, similarity: {sim}]")
        for i in current_paragraph:
            print(i)

    # LSI
    # Repeat above steps for LSI
    print(f"\n------------------LSI-------------------\n")
    lsi_model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=100)
    lsi_query = lsi_model[tfidf_query]

    lsi_corpus = lsi_model[corpus]

    lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus)

    # print(sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3])
    # print(lsi_model.show_topics())
    doc2similarity_lsi = enumerate(lsi_index[lsi_query])

    doc2similarity_lsi_sorted = sorted(doc2similarity_lsi, key=lambda kv: -kv[1])[:3]
    for parnum, sim in doc2similarity_lsi_sorted[:3]:
        current_paragraph = paragraphs[parnum].split("\n")[:5] if len(paragraphs[parnum].split("\n")) > 5 else paragraphs[parnum].split("\n")
        print(f"\n[paragraph: {parnum}, similarity: {sim}]")
        for i in current_paragraph:
            print(i)



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
    task_3()
    print("----------- Finished -----------")

    print()

    print("----------- Starting -----------")
    print("Task 4")
    #task_4()
    print("----------- Finished -----------")
