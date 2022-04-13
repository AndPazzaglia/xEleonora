#%%
import os
import pickle 
import numpy as np
import random

def EvalModel(model, author_name, input_words, out_lines_number, temperature):
    input_size = 20

    #%% load tokenizer        
    tok_path = os.path.join("tokenization", "tokenizer.pkl")
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)

    onehot_path = os.path.join("tokenization", "onehotencoder.pkl")
    with open(onehot_path, 'rb') as f:
        onehot = pickle.load(f)

    input_aut = onehot.transform(np.array([author_name]).reshape(1, -1))

    #%% model prediction        
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
        
    return final_output