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
    return list_of_bow

def task_1():
    with codecs.open("pg3300.txt", "r", "utf-8") as f:
        paragraphs = load_paragraphs(f)
        tokenized_paragraps = tokenize_paragraphs(paragraphs)
        stemmed_word_paragraphs = stem_words(tokenized_paragraps)
        return stemmed_word_paragraphs

def task_2():
    print("Task 2\n")
    words = task_1()
    with codecs.open("stop_words.txt", "r", "utf-8") as f:
        stop_words = f.read().split(",")
        corpus = create_corpus(stop_words, words)
        return corpus


if __name__ == '__main__':
    task_1()
    task_2()