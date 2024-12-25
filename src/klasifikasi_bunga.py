import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# Judul aplikasi Streamlit
st.title("Mengidentifikasi Jenis Tanaman Bunga Hias")

# Fungsi untuk memuat dan memproses gambar
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")  # Menambahkan konversi gambar ke mode RGB
    img = img.resize((150, 150))  # Sesuaikan dengan ukuran input model (150x150)
    img_array = np.array(img) / 255.0  # Normalisasi pixel ke rentang [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    return img_array

# Fungsi untuk melakukan prediksi dengan model
def predict_image(img):
    # Kelas yang diprediksi oleh model (sesuaikan dengan kelas yang ada di proyek kamu)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # Preprocessing gambar
    img_array = preprocess_image(img)
    
    # Memuat model yang telah disimpan
    model = tf.keras.models.load_model("E:/KULIAH/MY DOCS/SEMESTER 7/Machine Learning/UAP_umi/src/model_vgg/model_vgg.h5" )
    
    # Melakukan prediksi
    prediction = model.predict(img_array)
    
    # Menghitung probabilitas dan hasil prediksi
    predicted_class = class_names[np.argmax(prediction)]
    probability = np.max(tf.nn.softmax(prediction[0]))  # Probabilitas kelas terpilih
    
    return predicted_class, probability

# Bagian untuk mengunggah gambar
upload = st.file_uploader("Unggah gambar (Rock, Paper, Scissors)", type=['jpg', 'png', 'jpeg'])

if st.button("Prediksi", type="primary"):
    if upload is not None:
        st.image(upload, caption="Gambar yang diunggah", use_container_width=True)
        st.subheader("Hasil prediksi:")
        
        # Menampilkan progress bar dan prediksi
        with st.spinner('Memproses gambar untuk prediksi...'):
            predicted_class, probability = predict_image(upload)
        
        st.write(f"Prediksi: {predicted_class}")
        # st.write(f"Probabilitas: {probability * 100:.2f}%")
    else:
        st.write("Unggah gambar terlebih dahulu!")
