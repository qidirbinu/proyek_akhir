import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# Muat daftar stopwords bahasa inggris
stop_words = set(stopwords.words('english'))

# Fungsi untuk membersihkan teks
def preprocess_text(text):
    # Menghapus tanda baca dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Muat model
model = joblib.load('bullying_model.pkl')

# Fungsi prediksi
def predict_bullying(text):
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])
    return 'Cyberbullying' if prediction[0] == 1 else 'Not Cyberbullying'

# Aplikasi Streamlit
st.title("Cyberbullying Detection App")
st.write("Masukkan tweet untuk memprediksi apakah itu merupakan tweet bullying atau bukan.")

input_text = st.text_area("Input Tweet")
if st.button("Prediksi"):
    result = predict_bullying(input_text)
    st.write(f"Hasil Prediksi: {result}")
