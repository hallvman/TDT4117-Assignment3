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

f = codecs.open("pg3300.txt", "r", "utf-8")
stemmer = PorterStemmer()
translator = str.maketrans(
            '',
            '',
            string.punctuation + "\n\r\t")


# 1. Data loading and preprocessing
texts = f.read().split('\r\n\r\n')

print("Text:", len(texts))

paragraphs = []
for text in texts:
    if text != "":
        # 1.3
        if "gutenberg" not in text.lower():
            paragraphs.append(text)

#print("Paragraphs")
print("Paragraphs:", paragraphs[3])

# 1.4 tokenize paragraphs and remove punctuation
words = [None for i in range(len(paragraphs))]
for i in range(len(paragraphs)):
    words[i] = paragraphs[i].lower().translate(translator).split()


#print("Processed_words")
print("Processed_words:", words[3])

# 1.6  Using portstemmer stem words
stemmed_words = []
for i in range(len(words)):
    words[i] = [stemmer.stem(word) for word in words[i]]


#print("Stemmed_words")
print("Stemmed_words:", words[3])





