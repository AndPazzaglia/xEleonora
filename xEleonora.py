#%%

import pickle 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path

#%% open poetries file and keep only authors with more than 5 poetries

with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\poestries_dict.pkl', 'rb') as f:
    poetries_dict = pickle.load(f)

author_list = ['caio-valerio-catullo', 'dante-alighieri', 'dino-buzzati', 'niccolo-ugo-foscolo', 'giosue-carducci', 'gabriele-d-annunzio', 'giacomo-leopardi', 'alda-merini', 'eugenio-montale','pablo-neruda', 'giovanni-pascoli', 'cesare-pavese', 'luigi-pirandello']
poetries = []
for key in poetries_dict:
    if key in author_list:
        for p in poetries_dict[key]:
            poetries.append(p)

print('-------------------------')
print('Totale poesie processate: {}'.format(len(poetries)))
print('Lista autori:')
for author in author_list:
    print(author.replace('-', ' ').upper())
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

tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(poetries)
encoded = tokenizer.texts_to_sequences(poetries)

with open(r'C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\tokenizer.pkl', 'wb') as fp:
    pickle.dump(tokenizer, fp)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

input_size = 10
X = []
y = []
for seq in encoded:
    for i in range(input_size, len(seq)):
        X.append(seq[i-input_size:i])
        y.append(seq[i])

X = np.array(X)
y = np.array(y)

#%% define model

if path.exists(r"C:\Users\apazzaglia00\Documents\data science\ad Eleonora\lstm\xEleonora_model_00.h5"):
    # find up to date pre-trained model
    list_dir = os.listdir()
    dump_number = 0
    for file_name in list_dir:
        if "xEleonora_model_" in file_name:
            tmp = int(file_name[16:18])
            if tmp > dump_number:
                dump_number = tmp
    model_name = "xEleonora_model_" + str(dump_number).zfill(2) + ".h5"
    
    # load up to date pre-trained model
    model = load_model(model_name)
    history = model.fit(X, y, epochs=50, verbose=1, batch_size=128)
    dump_number = dump_number + 1
    model_name = "xEleonora_model_" + str(dump_number).zfill(2) + ".h5"
    model.save(model_name)
    
else:    
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 24, input_length=input_size))
    model.add(LSTM(32))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    
    # compile network
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    
    # fit network
    history = model.fit(X, y, epochs=50, verbose=1, batch_size=128, validation_split=0.2)
    model.save("xEleonora_model_00.h5")

plt.plot(history.history['sparse_categorical_accuracy'])
plt.show()



# %%
