import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Judul aplikasi Streamlit
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stFileUploader label {
        font-size: 18px;
        font-weight: bold;
    }
    .stSelectbox > div {
        font-size: 16px;
    }
    .stSpinner > div {
        color: #ff6347;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌼 Mengidentifikasi Jenis Tanaman Bunga Hias")

# Fungsi untuk memuat dan memproses gambar def preprocess_image(image, model_type):
def preprocess_image(image, model_type):
    img = Image.open(image)
    img = img.convert("RGB")  # Menambahkan konversi gambar ke mode RGB
    img = img.resize((150, 150))  # Sesuaikan dengan ukuran input model (150x150)
    img_array = np.array(img)

    if model_type == "GoogleNet":
        img_array = preprocess_input(img_array)
    elif model_type == "VGG16":
        img_array = img_array / 255.0  # Normalisasi pixel ke rentang [0, 1]

    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    return img_array

# Fungsi untuk melakukan prediksi dengan model
def predict_image(img, model_type):
    # Kelas yang diprediksi oleh model (sesuaikan dengan kelas yang ada di proyek kamu)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    # Preprocessing gambar
    img_array = preprocess_image(img, model_type)

    # Memuat model yang telah disimpan
    model_path = ""
    if model_type == "GoogleNet":
        model_path = "E:/KULIAH/MY DOCS/SEMESTER 7/Machine Learning/UAP_umi/src/model_ggnet/model_ggnet.h5"
    elif model_type == "VGG16":
        model_path = "E:/KULIAH/MY DOCS/SEMESTER 7/Machine Learning/UAP_umi/src/model_vgg/model_vgg.h5"

    model = tf.keras.models.load_model(model_path)

    # Melakukan prediksi
    prediction = model.predict(img_array)

    # Menghitung probabilitas dan hasil prediksi
    predicted_class = class_names[np.argmax(prediction)]
    probability = np.max(tf.nn.softmax(prediction[0]))  # Probabilitas kelas terpilih

    return predicted_class, probability

# Bagian untuk mengunggah gambar
st.sidebar.markdown("## 📸 Pilih Metode Input Gambar")
input_method = st.sidebar.radio("Pilih metode input:", ["Unggah Gambar", "Gunakan Kamera"])

if input_method == "Unggah Gambar":
    upload = st.file_uploader("Unggah gambar (JPG, PNG, JPEG):", type=['jpg', 'png', 'jpeg'])
elif input_method == "Gunakan Kamera":
    upload = st.camera_input("Ambil gambar dengan kamera")

# Pilihan model
model_type = st.selectbox("Pilih model untuk prediksi:", ["GoogleNet", "VGG16"])

if st.button("Prediksi", type="primary"):
    if upload is not None:
        st.image(upload, caption="Gambar yang diunggah", use_container_width=True)
        st.subheader("Hasil prediksi:")

        # Menampilkan progress bar dan prediksi
        with st.spinner('Memproses gambar untuk prediksi...'):
            predicted_class, probability = predict_image(upload, model_type)

        st.write(f"Prediksi: *{predicted_class}*")
        st.write(f"Probabilitas: *{probability * 100:.2f}%*")
    else:
        st.warning("Silakan unggah gambar atau ambil gambar dengan kamera terlebih dahulu!")