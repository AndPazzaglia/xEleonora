#%%
import os
import pickle 
from tensorflow.keras.models import load_model
import numpy as np
import random

#%% load model


model_name = "xEleonora_model_v3_20.h5"
model_path = os.path.join('models', model_name)
model = load_model(model_name)
input_size = 10

#%% load tokenizer
    
tok_path = os.path.join("tokenization", "tokenizer.pkl")
with open(tok_path, 'rb') as f:
    tokenizer = pickle.load(f)

onehot_path = os.path.join("tokenization", "onehotencoder.pkl")
with open(onehot_path, 'rb') as f:
    onehot = pickle.load(f)
    

#%% author choice

# author_name = 'ALDA MERINI'
author_name = 'CAIO VALERIO CATULLO'
# author_name = 'DINO BUZZATI'
# author_name = 'WILLIAM WORDSWORTH'
# author_name = 'EDGAR ALLAN POE'
# author_name = 'EUGENIO MONTALE'
# author_name = 'GIACOMO LEOPARDI'
# author_name = 'GIUSEPPE UNGARETTI'
# author_name = 'DANTE ALIGHIERI'
# author_name = 'WILLIAM SHAKESPEARE'

input_aut = onehot.transform(np.array([author_name]).reshape(1, -1))

#%% model prediction
    
out_lines_number = 14
temperature = 0.2
input_words = "Il tuo sorriso"
input_seq = np.array(tokenizer.texts_to_sequences([input_words]))
if len(input_seq[0]) > input_size:
    input_seq = input_seq[0][len(input_seq[0])-input_size:]
    input_seq = input_seq.reshape(1, input_size)
elif len(input_seq[0]) < input_size:
    input_seq = np.append(np.zeros(input_size-len(input_seq[0])), input_seq[0])
    input_seq = input_seq.reshape(1, input_size)

def predict_word(input_seq, model, tokenizer, temperature):
    preds = model.predict([input_seq, input_aut])
    weights = preds**(1/temperature) / np.sum(preds**(1/temperature))
    weights = weights.reshape(len(tokenizer.word_index)+1)
    word_id = random.choices(population=np.arange(0, 1 + len(tokenizer.word_index)), weights=weights)
    return word_id[0]

lines = 0
output_words = input_words
for i in range(500):
    word_id = predict_word(input_seq, model, tokenizer, temperature)
    out_word = tokenizer.index_word[word_id]
    if out_word == '\n':
        lines = lines + 1
        if lines >= out_lines_number:
            break
    output_words = output_words + ' ' + out_word
    tmp = input_seq[0][1:]
    input_seq = np.reshape(np.append(tmp, word_id), (1, input_size))
    
final_output = output_words.replace("\n ", "\n").replace(" .", ".").replace(" ,", ",")
print('--------------------------------------------------------')
print(final_output)
print('--------------------------------------------------------')
    
# text_number = 0
# list_dir = os.listdir()
# for file_name in list_dir:
#     if "text_" in file_name:
#         tmp = int(file_name[5:7])
#         if tmp > text_number:
#             text_number = tmp
#%% text to speech

# from gtts import gTTS
# from playsound import playsound
# import os

# tts = gTTS(text=final_output.replace(' l ', ' il '), lang='it')

# audio_number = 0
# list_dir = os.listdir()
# for file_name in list_dir:
#     if "audio_" in file_name:
#         tmp = int(file_name[6:8])
#         if tmp > audio_number:
#             audio_number = tmp

# audio_name = 'audio_' + str(audio_number + 1).zfill(2) + '.mp3'
# tts.save(audio_name)
# playsound(audio_name)