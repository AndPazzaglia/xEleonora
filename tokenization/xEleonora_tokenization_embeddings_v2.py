#%%
import os
import pickle 
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import KeyedVectors

#%% load pre trained word embeddings
word_vectors = KeyedVectors.load("W2V.kv", mmap='r+')
vocabs = word_vectors.index_to_key
vectors = word_vectors.vectors

#%% open poetries file and keep only authors with more than 5 poetries

# filename = os.path.join()
with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\poestries_dict.pkl', 'rb') as f:
    poetries_dict = pickle.load(f)

poetries = []
author_list = []
for key in poetries_dict:
    for p in poetries_dict[key]:
        poetries.append(p)
        author_list.append(key.replace('-', ' ').upper())

print('-------------------------')
print('Totale poesie processate: {}'.format(len(poetries)))
print('Lista autori:')
for author in set(author_list):
    print(author)
print('-------------------------')

#%% prepare dataset

table = str.maketrans('', '', '!"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~')
for i in range(len(poetries)):
    poetries[i] = poetries[i].lower()
    poetries[i] = poetries[i].replace("\r", "")
    poetries[i] = poetries[i].replace("\n", " \n ")
    poetries[i] = poetries[i].replace("  ", " ")
    poetries[i] = poetries[i].replace("’", " ")
    poetries[i] = poetries[i].replace(",", " ,")
    poetries[i] = poetries[i].replace(".", " .")
    poetries[i] = poetries[i].translate(table)
poetries = np.array(poetries)
authors = np.array(author_list)

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(poetries)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

#%%
import difflib
embedding_dim = 300
hits = 0
misses = 0

# look for missed and hit words
missed_words = []
hit_words = []
for word, i in tokenizer.word_index.items():
    embedding_index = word_vectors.get_index(word, default=-1)
    if embedding_index != -1:
        hits += 1
        hit_words.append(word)
    else:
        misses += 1
        missed_words.append(word)

print("Found %d words (%d misses)" % (hits, misses))

#%% deal with missed words - find similar words in hit words from pre trained word embeddings
from tqdm import tqdm

poetries = poetries.tolist()
missed_words.pop(0) # remove \n from missed words
for missed in tqdm(missed_words):
    # simil_word = difflib.get_close_matches(missed, hit_words, n=1)
    simil_word = difflib.get_close_matches(missed, vocabs, n=1)
    if len(simil_word) > 0:
        simil_word = simil_word[0]
    else:
        simil_word = '.'
    # add spaces to missed and similar words to assure that only whole word is replaced
    missed = ' ' + missed + ' '
    simil_word = ' ' + simil_word + ' '
    # replace whole words
    poetries = [p.replace(missed, simil_word) for p in poetries]

clean_poetries_authors = np.array([authors, np.array(poetries)])
with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\clean_poetries_authors.pkl', 'wb') as fp:
    pickle.dump(clean_poetries_authors, fp)

#%% final fit of tokenizer and embeddings matrix
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(poetries)

with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\tokenizer.pkl', 'wb') as fp:
    pickle.dump(tokenizer, fp)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

missed_words = []
hit_words = []
hits = 0
misses = 0
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_index = word_vectors.get_index(word, default=-1)
    if embedding_index != -1:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = word_vectors.vectors[i,:]
        hits += 1
        hit_words.append(word)
    else:
        misses += 1
        missed_words.append(word)

print("Converted %d words (%d misses)" % (hits, misses))

with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\embedding_matrix.pkl', 'wb') as fp:
    pickle.dump(embedding_matrix, fp)

# %%
