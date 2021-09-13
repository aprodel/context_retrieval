
import fire
import json
import numpy as np
import re
import math
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

nltk.download('stopwords')
nltk.download('punkt')

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_jsonl(input_path):
    
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def convert_lower_case(data):
    return np.char.lower(data)

def remove_foreign_chars(data):
    return re.sub("[^a-zA-Z0-9\s]+", "", data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = np.char.replace(data, "'", "")
    return data

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = np.array([remove_foreign_chars(text) for text in data])
    data = np.array([remove_stop_words(text) for text in data])
    data = np.array([convert_numbers(text) for text in data])
    data = np.array([stemming(text) for text in data])
    data = remove_punctuation(data) 
    #data = remove_stop_words(data) 
    return data

def preprocess_from_file(path):
	"""
    Loads, preprocess and stores contexts from data file
    :param path: path to JSONL data file
    """

	print("Loading data...")
	data = load_jsonl(path)
	data = np.array([elt["passage"] for elt in data])
	np.save("./loaded_contexts.npy", data)
	np.save("./len_data.npy", len(data))
	print("Preprocessing...")
	result = preprocess(data)
	print("Saving data...")
	np.save("./loaded_data.npy", result)

	return


def doc_freq(DF, word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


def compute_TFIDF():
	"""
    Loads preprocessed data, compute document frequencies and TF-IDF matrix on context dataset
    """

	train_preprocessed = np.load("./loaded_data.npy")

	DF = {}
	for i in range(len(train_preprocessed)):
	    text = train_preprocessed[i].split(" ")
	    for word in text:
	        if word != "":
	            try:
	                DF[word].add(i)
	            except:
	                DF[word] = {i}

	for i in DF:
	    DF[i] = len(DF[i])
	    
	total_vocab_size = len(DF)
	total_vocab = [x for x in DF]


	doc = 0

	tf_idf = {}

	for i in range(len(train_preprocessed)):
	    
	    tokens = train_preprocessed[i].split(" ")
	    
	    counter = Counter(tokens)
	    words_count = len(tokens)
	    
	    for token in np.unique(tokens):
	        
	        tf = counter[token]/words_count
	        df = doc_freq(DF, token)
	        idf = np.log((len(train_preprocessed)+1)/(df+1))
	        
	        tf_idf[doc, token] = tf*idf

	    doc += 1

	D = np.zeros((len(train_preprocessed), total_vocab_size))
	for i in tf_idf:
	    try:
	    	ind = total_vocab.index(i[1])
	    	D[i[0]][ind] = tf_idf[i]
	    except:
	    	pass

	save_obj(DF, "./DF.pkl")
	np.save("./TFIDF_matrix.npy", D)

	return


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def gen_vector(DF, tokens, N):

	total_vocab_size = len(DF)
	total_vocab = [x for x in DF]

	Q = np.zeros((len(total_vocab)))

	counter = Counter(tokens)
	words_count = len(tokens)

	query_weights = {}

	for token in np.unique(tokens):
	    
	    tf = counter[token]/words_count
	    df = doc_freq(DF, token)
	    idf = math.log((N+1)/(df+1))

	    try:
	        ind = total_vocab.index(token)
	        Q[ind] = tf*idf
	    except:
	        pass
	return Q




def find_context(question, k):
	"""
	Finds best contexts among suggested dataset.
	:param question: input question for context retrieval (string)
	:param k: number of contexts to retrieve
	"""

	DF = load_obj("./DF.pkl")
	D = np.load("./TFIDF_matrix.npy")
	N = np.load("./len_data.npy")
	contexts = np.load("./loaded_contexts.npy")

	preprocessed_query = preprocess(np.array([question]))[0]
	tokens = word_tokenize(str(preprocessed_query))

	d_cosines = []

	query_vector = gen_vector(DF, tokens, N)

	for d in D:
	    d_cosines.append(cosine_sim(query_vector, d))
	    
	return contexts[np.array(d_cosines).argsort()[-k:][::-1]]



if __name__ == "__main__":
    fire.Fire({
        "preprocess": preprocess_from_file,
        "compute_TFIDF": compute_TFIDF,
        "find_context": find_context
    })


