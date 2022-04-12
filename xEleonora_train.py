#%%

import pickle 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU, concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path

#%% load dataset, tokenizer, embeddings matrix

auth_path = os.path.join('tokenization','clean_poetries_authors.pkl')
with open(auth_path, 'rb') as f:
    clean_poetries_authors = pickle.load(f)

tok_path = os.path.join('tokenization', 'tokenizer.pkl')
with open(tok_path, 'rb') as f:
    tokenizer = pickle.load(f)

emb_path = os.path.join('tokenization', 'embedding_matrix.pkl')
with open(emb_path, 'rb') as f:
    embedding_matrix = pickle.load(f)

#%% prepare dataset: add one hot encoding of authors

from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder(sparse=False)
authors = clean_poetries_authors[0, :]
authors = authors.reshape(-1, 1)
authors_onehot = onehot.fit_transform(authors)
authors_number = len(authors_onehot[0])

onehot_path = os.path.join('tokenization', 'onehotencoder.pkl')
with open(onehot_path, 'wb') as fp:
    pickle.dump(onehot, fp)

#%% prepare dataset: convert poetries to sequences
from sklearn.model_selection import train_test_split
poetries = clean_poetries_authors[1, :]
encoded = tokenizer.texts_to_sequences(poetries)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

input_size = 10
X = []
y = []
y_emb = []
for j, seq in enumerate(encoded):
    aut_oh = authors_onehot[j,:]
    for i in range(input_size, len(seq)):
        tmp = np.array(seq[i-input_size:i])
        tmp = np.concatenate((tmp, aut_oh))
        X.append(tmp)
        y.append(seq[i])
        y_emb.append(embedding_matrix[seq[i]])

X = np.array(X)
y = np.array(y)
y_emb = np.array(y_emb)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_emb_train, y_emb_test = train_test_split(X, y_emb, test_size=0.2, random_state=42)

X_train_seq = X_train[:, 0:input_size]
X_train_aut = X_train[:, input_size:]
X_test_seq = X_test[:, 0:input_size]
X_test_aut = X_test[:, input_size:]

#%% define model new structure - regression of embeddings & authors one hot encoding

# input_seq = Input(shape=(input_size,))
# emb = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=input_size, trainable=False)(input_seq)
# gru = GRU(128)(emb)

# input_aut = Input(shape=(authors_number,))
# conc = concatenate([gru, input_aut])

# dense1 = Dense(256, activation='relu')(conc)
# dense2 = Dense(256, activation='relu')(dense1)
# out = Dense(300, activation='linear')(dense2)

# model = Model(inputs=[input_seq, input_aut], outputs=out)

# print(model.summary())

# # compile network
# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# # fit network
# history = model.fit([X_train_seq, X_train_aut], y_emb_train, epochs=10, verbose=1, batch_size=256, validation_data=([X_test_seq, X_test_aut], y_emb_test))
# model.save("xEleonora_model_v2_20.h5")

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.show()

#%% define model new structure - cross entropy & authors one hot encoding

input_seq = Input(shape=(input_size,))
emb = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=input_size, trainable=False)(input_seq)
gru = GRU(128)(emb)

input_aut = Input(shape=(authors_number,))
conc = concatenate([gru, input_aut])

dense1 = Dense(128, activation='relu')(conc)
dense2 = Dense(128, activation='relu')(dense1)
out = Dense(vocab_size, activation='softmax')(dense2)

model = Model(inputs=[input_seq, input_aut], outputs=out)

print(model.summary())

# compile network
opt = Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])

# define checkpoint saving callback
checkpoint_saving = ModelCheckpoint(
    'models\epoch_{epoch}', save_freq='epoch',
)

# fit network
history = model.fit(
    [X_train_seq, X_train_aut], y_train, epochs=50, verbose=1, 
    batch_size=64, validation_data=([X_test_seq, X_test_aut], y_test),
    callbacks=[checkpoint_saving, TensorBoard()])


# plt.figure()
# plt.plot(history.history['sparse_categorical_accuracy'])
# plt.plot(history.history['val_sparse_categorical_accuracy'])
# plt.show()
