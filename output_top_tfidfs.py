import sys
import string
from math import log
from os import listdir
from os.path import isfile, isdir, join

#variables initialization

feature_type = sys.argv[1]
num_features = int(sys.argv[2])
d = './data/'
catdirs = [join(d,o) for o in listdir(d) if isdir(join(d,o))] #categories division

def contain_punctuation(s):
    punctuation = [c for c in string.punctuation]
    punctuation.append(' ')
    r = any(c in s for c in punctuation) 
    return r

def normalise_tfs(tfs,total):
    for k,v in tfs.items():
        tfs[k] = v / total
    return tfs

def log_idfs(idfs,num_cats):  #This reflects the importance of a term in a document relative to all documents in the corpus
    for k,v in idfs.items():
        idfs[k] = log(num_cats / v)
    return idfs 

#lists initialization

cat_tfs = {}
cat_tf_idfs = {}
idfs = {}

for cat in catdirs: #iterate into every category, so we start from one class, after cycling we go to the other
    print(cat)
    tfs = {}
    sum_freqs = 0
    ngramorwords = [join(cat,f) for f in listdir(cat) if isfile(join(cat, f)) and feature_type in f]
    #this guy here creates a directory to linear.words and takes into consideration the amount of word that we declared for each class

    for wordfile in ngramorwords: #for every element in linear.words
        f = open(wordfile,'r')
        for l in f:
            l = l.rstrip() #we take exactly the position of the word with its relative count
            ngram = '\t'.join(i for i in l.split('\t')[:-1]) #we only get the word
            freq = int(l.split('\t')[-1]) #we only get the relative count
            tfs[ngram] = freq
            sum_freqs+=freq #here we sum all the frquencies of all the words
            if ngram in idfs:
                idfs[ngram]+=1
            else:
                idfs[ngram]=1
        f.close()

    tfs = normalise_tfs(tfs,sum_freqs) #frequencies count normalization, really? Yes, all of the preceedent steps have been done to get here
    cat_tfs[cat] = tfs #ordered vocabulary with normalized data

    #for k in sorted(idfs, key=tfs.get, reverse=True)[:10]:
    #    print(k,idfs[k])

idfs = log_idfs(idfs, len(catdirs)) #weight of every word in the data with idfs

vocab=[]

for cat in catdirs:
    tf_idfs = {}
    tfs = cat_tfs[cat]
    for ngram,tf in tfs.items(): #tf takes only the normalized wordcount
        tf_idfs[ngram] = tf * idfs[ngram]
    cat_tf_idfs[cat] = tf_idfs #term frequency-inverse document frequency

    c = 0
    for k in sorted(tf_idfs, key=tf_idfs.get, reverse=True):
        #only keep top k dimensions per category. Also, we won't keep ngrams with spaces
        if c == num_features:
            break
        if k not in vocab and not contain_punctuation(k):
            vocab.append(k)
            c+=1

#This is the end of the code

print("VOCAB SIZE:",len(vocab))

#Write tf-idfs for each category
for cat in catdirs:
    tf_idfs = cat_tf_idfs[cat]
    f = open(join(cat,'tf_idfs.txt'),'w')
    for ngram in sorted(vocab):
        if ngram in tf_idfs:
            f.write(ngram+' '+str(tf_idfs[ngram])+'\n')
        else:
            f.write(ngram+' 0.0\n')
    f.close()
 
#write the vocab
vocab_file = open("./data/vocab_file.txt",'w')
for ngram in sorted(vocab):
    vocab_file.write(ngram+'\n')
vocab_file.close()

