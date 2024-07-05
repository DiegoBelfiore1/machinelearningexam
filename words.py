import string
import sys
from os import listdir
from os.path import isfile, isdir, join
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from collections import defaultdict, Counter
from textblob import TextBlob
from functionwords import FunctionWords

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def contain_symbols(s):
    symbols = [c for c in string.punctuation]
    symbols.extend([d for d in string.digits])
    return any(c in s for c in symbols)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

d = './data/'
catdirs = [join(d, o) for o in listdir(d) if isdir(join(d, o))]

# Initialize FunctionWords
fw = FunctionWords(function_words_list='english')

# Define filler words
filler_words = set(['uh', 'um', 'well', 'actually', 'literally'])

# Dictionaries to store counts and scores
class_word_counts = {}
class_pos_counts = defaultdict(Counter)
class_polarities = defaultdict(list)
class_function_word_counts = defaultdict(Counter)
class_filler_word_counts = defaultdict(Counter)

lemmatizer = WordNetLemmatizer()

for cat in catdirs:
    words = {}
    class_name = cat.split('/')[-1]
    f = open(join(cat, 'linear.txt'), 'r')
    for l in f:
        l = l.rstrip('\n').lower()
        tokens = l.split()
        for word in tokens:
            if contain_symbols(word):
                continue
            lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            if lemmatized_word in words:
                words[lemmatized_word] += 1
            else:
                words[lemmatized_word] = 1
            
            # POS tagging and counting
            pos_tag = nltk.pos_tag([lemmatized_word])[0][1]
            class_pos_counts[class_name][pos_tag] += 1

        # Function words count
        function_words_counts = fw.transform(' '.join(tokens))
        for feature_name, count in zip(fw.get_feature_names(), function_words_counts):
            class_function_word_counts[class_name][feature_name] += count

        # Filler words count
        filler_counts = Counter(token for token in tokens if token in filler_words)
        class_filler_word_counts[class_name].update(filler_counts)
        
        # Calculate polarity for the line and add to the class polarity list
        blob = TextBlob(l)
        class_polarities[class_name].append(blob.sentiment.polarity)

    f.close()

    wordfile = open(join(cat, "linear.words"), 'w')
    for k in sorted(words, key=words.get, reverse=True):
        wordfile.write(k + '\t' + str(words[k]) + '\n')
    wordfile.close()

    # Sum up the word counts for this class
    total_count = sum(words.values())
    class_word_counts[class_name] = total_count

# Print word counts, POS analysis, polarity analysis, function words, and filler words
for class_name, total_count in class_word_counts.items():
    print(f"Class '{class_name}': Total Word Occurrences = {total_count}")
    print(f"POS Analysis for Class '{class_name}':")
    for pos_tag, count in class_pos_counts[class_name].items():
        print(f"  {pos_tag}: {count}")
    
    # Calculate and print the average polarity for the class
    avg_polarity = sum(class_polarities[class_name]) / len(class_polarities[class_name])
    print(f"Average Polarity for Class '{class_name}': {avg_polarity}")

    # Calculate total function words and filler words
    total_function_words = sum(class_function_word_counts[class_name].values())
    total_filler_words = sum(class_filler_word_counts[class_name].values())

    # Calculate percentages
    function_words_percentage = (total_function_words / total_count) * 100
    filler_words_percentage = (total_filler_words / total_count) * 100

    # Print percentages
    print(f"Function Words in Class '{class_name}': {function_words_percentage:.2f}%")
    print(f"Filler Words in Class '{class_name}': {filler_words_percentage:.2f}%")
    print()
