import re
import string
import sys
import numpy as np
from math import isnan
from os import listdir
from os.path import isfile, isdir, join
import nltk
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from functionwords import FunctionWords
from textblob import TextBlob

# Initialize FunctionWords
fw = FunctionWords(function_words_list='english')
class_function_word_counts = defaultdict(Counter)

# Ensure NLTK resources are available
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

feature_type = sys.argv[1]

# Predefined lists for POS and filler words
pos_tags = {
    'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
    'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'prepositions': ['IN'],
    'possessive_pronouns': ['PRP$', 'WP$'],
    'adjectives': ['JJ', 'JJR', 'JJS']
}
filler_words = ['uh', 'um', 'well', 'actually', 'literally']

def read_vocab():
    with open('./data/vocab_file.txt', 'r') as f:
        vocab = f.read().splitlines()
    return vocab

def get_words(l):
    l = l.lower()
    words = {}
    for word in l.split():
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words

def get_ngrams(l, n):
    l = l.lower()
    ngrams = {}
    for i in range(0, len(l) - n + 1):
        ngram = l[i:i + n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams

def normalise(v):
    return v / sum(v)

def clean_docs(d, docs):
    m = []
    retained_docs = []
    for url in docs:
        if not isnan(sum(d[url])) and sum(d[url]) != 0:
            m.append(d[url])
            retained_docs.append(url)
    return np.array(m), retained_docs

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in ["...", "’", "n't", "'s", "'m", "'re", "'ve"]]
    return tokens

def calculate_pos_filler_averages(text):
    tokens = preprocess(text)
    tagged = nltk.pos_tag(tokens)
    
    pos_counts = {key: 0 for key in pos_tags}
    filler_count = 0
    
    for word, tag in tagged:
        for pos, tags in pos_tags.items():
            if tag in tags:
                pos_counts[pos] += 1
        if word in filler_words:
            filler_count += 1
    
    total_words = len(tokens)
    pos_averages = {pos: count / total_words for pos, count in pos_counts.items()}
    filler_average = filler_count / total_words
    
    return pos_averages, filler_average

def calculate_function_word_average(text):
    tokens = preprocess(text)
    function_words_counts = fw.transform(' '.join(tokens))
    
    function_word_average = sum(function_words_counts) / len(tokens) if tokens else 0
    return function_word_average

def calculate_average_polarity(text):
    blob = TextBlob(text)
    sentence_polarities = [sentence.sentiment.polarity for sentence in blob.sentences]
    average_polarity = sum(sentence_polarities) / len(sentence_polarities) if sentence_polarities else 0
    return average_polarity

d = './data'
catdirs = [join(d, o) for o in listdir(d) if isdir(join(d, o))]
vocab = read_vocab()

for cat in catdirs:
    print(cat)
    url = ""
    docs = []
    vecs = {}
    pos_filler_averages = {}
    function_word_averages = {}
    polarity_averages = {}
    
    doc_file = open(join(cat, "linear.txt"), 'r')
    for l in doc_file:
        l = l.rstrip('\n')
        if l[:4] == "<doc":
            m = re.search("date=(.*)>", l)
            url = m.group(1).replace(',', ' ')
            docs.append(url)
            vecs[url] = np.zeros(len(vocab))
            text = ""
        if l[:5] == "</doc":
            vecs[url] = normalise(vecs[url])
            pos_averages, filler_average = calculate_pos_filler_averages(text)
            function_word_average = calculate_function_word_average(text)
            average_polarity = calculate_average_polarity(text)
            pos_filler_averages[url] = (pos_averages, filler_average)
            function_word_averages[url] = function_word_average
            polarity_averages[url] = average_polarity
            print(url)
        else:
            text += l + " "
        
        if feature_type == "ngrams":
            for i in range(3, 7):  # hacky...
                ngrams = get_ngrams(l, i)
                for k, v in ngrams.items():
                    if k in vocab:
                        vecs[url][vocab.index(k)] += v
        if feature_type == "words":
            words = get_words(l)
            for k, v in words.items():
                if k in vocab:
                    vecs[url][vocab.index(k)] += v

    doc_file.close()
    m, retained_docs = clean_docs(vecs, docs)
    print("------------------")
    print("NUM ORIGINAL DOCS:", len(docs))
    print("NUM RETAINED DOCS:", len(retained_docs))
    
    vec_file = open(join(cat, "vecs.csv"), 'w')
    
    for i in range(len(retained_docs)):
        url = retained_docs[i]
        pos_averages, filler_average = pos_filler_averages[url]
        function_word_average = function_word_averages[url]
        average_polarity = polarity_averages[url]
        row = [url] + list(m[i]) + [
            pos_averages['nouns'],
            pos_averages['verbs'],
            pos_averages['prepositions'],
            pos_averages['possessive_pronouns'],
            pos_averages['adjectives'],
            filler_average,
            function_word_average,
            average_polarity
        ]
        vec_file.write(','.join(map(str, row)) + '\n')
    
    vec_file.close()
