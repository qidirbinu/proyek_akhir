import streamlit as st
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words

# Load the trained model
with open('bullying_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess the input (sesuaikan jika perlu)
def preprocess_text(input_text):
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)
    input_text = input_text.lower()
    input_text = ' '.join([word for word in text.split() if word not in stop_words])
    return np.array([input_text])

# Streamlit app
st.title('Text Classification App')
st.write('Masukkan kalimat dan dapatkan hasil prediksi golongan A atau B berdasarkan model')

# Input dari pengguna
user_input = st.text_input('Masukkan kalimat:', '')

if user_input:
    # Preprocess input
    processed_input = preprocess_text(user_input)
    
    # Prediksi menggunakan model
    prediction = model.predict(processed_input)
    prediction_prob = model.predict_proba(processed_input)[0][1]  # Mendapatkan probabilitas kelas B

    # Tentukan kategori berdasarkan threshold 0.5
    if prediction_prob > 0.5:
        result = "Cyberbullying"
    else:
        result = "bukan Cyberbullying"

    # Tampilkan hasil prediksi
    st.write(f"Prediksi: {result}")
    st.write(f"Probabilitas: {prediction_prob:.2f}")
