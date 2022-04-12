import streamlit as st
from tensorflow.keras.models import load_model
from utils.EvalModel import EvalModel
import os
import pickle

st.title('xEleonora')
st.write(
    'xEleonora is an AI model which creates poetries using the writing style from a preset list of authors.')
st.sidebar.header('Instructions:')
st.sidebar.write(
    'Fill the following inputs and then hit the Run button!')

#%% load model
model_name = "epoch_3"
model_path = os.path.join('models', model_name)
model = load_model(model_path)

#%% choice of the author style
auth_path = os.path.join('tokenization','clean_poetries_authors.pkl')
with open(auth_path, 'rb') as f:
    clean_poetries_authors = pickle.load(f)
authors = clean_poetries_authors[0, :]
authors = list(dict.fromkeys(authors))
selected_author = st.sidebar.selectbox('Select the author style', authors, index=1)

#%% text input
input_words = st.sidebar.text_input('Write initial words of the poetry', value="Il tuo sorriso")
out_lines_number = st.sidebar.slider('Select the number of lines of the poetry', min_value=1, max_value=50, value=13, step=1)
temperature = st.sidebar.slider('Select the model temperature (higher leads to more variable results)', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

if st.sidebar.button('Run model'):
    final_output = EvalModel(
        model, author_name=selected_author, input_words=input_words, out_lines_number=out_lines_number, temperature=temperature)
    st.text(final_output)