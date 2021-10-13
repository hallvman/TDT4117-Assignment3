# 1.0
import random; random.seed(123)
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
texts = f.read().split('\n\n')

paragraphs = []
for text in texts:
    if text != "":
        # 1.3
        if "gutenberg" not in text.lower():
            paragraphs.append(text)

print("Paragraphs")
#print("Paragraphs", paragraphs)

# 1.4 tokenize paragraphs
words = [None for i in range(len(paragraphs))]
for i, p in enumerate(paragraphs):
    paragraphs[i] = p.split(" ")
    words[i] = paragraphs[i]

print("Words")
#print("Words", words)

# 1.5 Remove punctuation from text
processed_words = []
for w in words:
    for i, word in enumerate(w):
        # Remove punctuations
        words[i] = word.lower().translate(translator)
        if p in words[i]:
            processed_words[i] = words[i]

print("Processed_words")
#print("Processed_words", processed_words)

# 1.6  Using portstemmer stem words
stemmed_words = []
for i, word in enumerate(processed_words):
    processed_words[i] = stemmer.stem(word)
    stemmed_words[i] = processed_words[i]

print(stemmed_words)





