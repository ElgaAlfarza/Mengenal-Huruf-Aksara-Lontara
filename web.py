import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load Model
MODEL_PATH = "Otaknya.h5"
model = load_model(MODEL_PATH)

# Load Class Indices
# Update dengan dictionary sesuai jumlah kelas dalam model Anda
CLASS_NAMES = {
    0: 'a',
    1: 'ba',
    2: 'ca',
    3: 'da',
    4: 'ga',
    5: 'ha',
    6: 'ja',
    7: 'ka',
    8: 'la',
    9: 'ma',
    10: 'mpa',
    11: 'na',
    12: 'nca',
    13: 'nga',
    14: 'ngka',
    15: 'nra',
    16: 'nya',
    17: 'pa',
    18: 'ra',
    19: 'sa',
    20: 'ta',
    21: 'wa',
    22: 'ya'
}

# Function to Predict
def predict_image(image):
    try:
        # Resize image sesuai input model
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
        
        # Prediksi menggunakan model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Error handling jika kelas tidak ditemukan
        if predicted_class not in CLASS_NAMES:
            return "Kelas tidak dikenali", 0.0

        return CLASS_NAMES[predicted_class], confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# Streamlit App
st.title("Pengenalan Huruf Aksara Lontara")
st.write("Unggah gambar aksara lontara untuk diklasifikasikan.")

# File Uploader
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Prediction
    if st.button("Klasifikasi"):
        predicted_class, confidence = predict_image(image)
        st.write(f"Prediksi: **{predicted_class}**")
        st.write(f"Tingkat Kepercayaan: **{confidence:.2f}**")
