import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
Stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return Stemmer.stem(word.lower())

def bag_of_word(tokenize_sentence, all_words):
     
    sentence_words = [stem(w) for w in tokenize_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

