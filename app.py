import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('bullying_model.pkl', 'wb') as file:
    model = pickle.load(file)

# Function to preprocess the input (sesuaikan jika perlu)
def preprocess_text(input_text):
    # Lakukan preprocessing pada input_text jika diperlukan, misalnya tokenization, stop word removal, dll.
    # Untuk contoh ini, kita asumsikan input sudah bisa langsung diprediksi oleh model
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
