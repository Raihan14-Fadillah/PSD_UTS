import streamlit as st
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from scipy.stats import kurtosis
from sklearn.decomposition import PCA

# Load the saved Min-Max scaler, KNN model, and PCA model
scaler = load('scaler_minmax.pkl')
knn_model = load('knn_model.pkl')

# Load the PCA model with 10 components
pca_model = load('pca_model.pkl')

# Tampilan Streamlit dalam bahasa Indonesia
st.title("Pengenalan Emosi dari Suara")
st.subheader("Raihan Fadillah - NIM 210411100003")

# Unggah file audio
uploaded_file = st.file_uploader("Unggah file audio (format .wav)", type=["wav"])

# Daftar lengkap label emosi
emojis = {
    'YAF_sad': 'Sedih',
    'YAF_pleasant_surprised': 'Senang Terkejut',
    'YAF_neutral': 'Netral',
    'YAF_happy': 'Senang',
    'YAF_fear': 'Takut',
    'YAF_disgust': 'Jijik',
    'YAF_angry': 'Marah',
    'OAF_Sad': 'Sedih',
    'OAF_Pleasant_surprise': 'Senang Terkejut',
    'OAF_neutral': 'Netral',
    'OAF_happy': 'Senang',
    'OAF_Fear': 'Takut',
    'OAF_disgust': 'Jijik',
    'OAF_angry': 'Marah'
}

if uploaded_file is not None:
    # Memuat dan memproses file audio yang diunggah
    audio_data, sr = librosa.load(uploaded_file)
    mean = np.mean(audio_data)
    std_dev = np.std(audio_data)
    max_value = np.max(audio_data)
    min_value = np.min(audio_data)
    median = np.median(audio_data)
    kurt = kurtosis(audio_data)  # Menghitung kurtosis langsung untuk audio_data
    skewness = kurtosis(audio_data)  # Menghitung skewness berdasarkan kurtosis
    
    q1 = np.percentile(audio_data, 25)
    q3 = np.percentile(audio_data, 75)
    mode_value = float(pd.Series(audio_data).mode().iloc[0])  # Perhitungan mode yang diperbaiki
    iqr = q3 - q1
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
    zcr_median = np.median(librosa.feature.zero_crossing_rate(y=audio_data))
    zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=audio_data))
    zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=audio_data)[0])
    zcr_skew = kurtosis(librosa.feature.zero_crossing_rate(y=audio_data)[0])  # Menghitung skewness berdasarkan kurtosis
    energy = np.sum(audio_data**2) / len(audio_data)
    energy_median = np.median(audio_data**2)
    energy_std_dev = np.std(audio_data**2)
    energy_kurtosis = kurtosis(audio_data**2)
    energy_skew = kurtosis(audio_data**2)  # Menghitung skewness berdasarkan kurtosis

    # Membuat DataFrame
    data = pd.DataFrame([[
        mean, std_dev, max_value, min_value, median, skewness, kurt, q1, q3, mode_value,
        iqr, zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew, energy,
        energy_median, energy_std_dev, energy_kurtosis, energy_skew
    ]], columns=[
        'Mean', 'Std Dev', 'Max', 'Min', 'Median', 'Skew', 'Kurtosis', 'Q1', 'Q3', 'Mode',
        'IQR', 'ZCR Mean', 'ZCR Median', 'ZCR Std Dev', 'ZCR Kurtosis', 'ZCR Skew', 'Energy',
        'Energy Median', 'Energy Std Dev', 'Energy Kurtosis', 'Energy Skew'
    ])

    # Normalisasi data menggunakan Min-Max scaler
    normalized_data = scaler.transform(data)
    
    # Transformasi fitur dengan model PCA
    # pca_features = pca_model.transform(normalized_data)

    # Prediksi label emosi menggunakan model KNN
    prediction = knn_model.predict(normalized_data) 

    # Tampilkan audio yang diunggah
    st.subheader("Audio yang Diunggah:")
    st.audio(uploaded_file, format="audio/wav")

    # Menampilkan label emosi yang diprediksi dalam bahasa Indonesia
    st.subheader("Emosi yang Diprediksi:")
    if prediction[0] in emojis:
        predicted_emotion = emojis[prediction[0]]
    else:
        predicted_emotion = "Tidak Diketahui"

    # Tampilkan tombol untuk mengungkapkan emosi yang diprediksi
    with st.expander("Lihat Emosi yang Diprediksi"):
        st.write(predicted_emotion)

    # Tampilkan nilai statistik dari file audio yang diunggah dalam ekspander
    stat_expander = st.expander("Statistik Audio:")
    stat_expander.write(f"Rata-rata (Mean): {mean:.4f}")
    stat_expander.write(f"Deviasi Standar (Std Dev): {std_dev:.4f}")
    stat_expander.write(f"Nilai Maksimum (Max): {max_value:.4f}")
    stat_expander.write(f"Nilai Minimum (Min): {min_value:.4f}")
    stat_expander.write(f"Median: {median:.4f}")
    stat_expander.write(f"Skewness: {skewness:.4f}")
    stat_expander.write(f"Kurtosis: {kurt:.4f}")
    stat_expander.write(f"Kuartil Pertama (Q1): {q1:.4f}")
    stat_expander.write(f"Kuartil Ketiga (Q3): {q3:.4f}")
    stat_expander.write(f"Mode: {mode_value:.4f}")
    stat_expander.write(f"Rentang Interkuartil (IQR): {iqr:.4f}")
    stat_expander.write(f"Zero Crossing Rate (Mean): {zcr_mean:.4f}")
    stat_expander.write(f"Zero Crossing Rate (Median): {zcr_median:.4f}")
    stat_expander.write(f"Zero Crossing Rate (Std Dev): {zcr_std_dev:.4f}")
    stat_expander.write(f"Zero Crossing Rate (Kurtosis): {zcr_kurtosis:.4f}")
    stat_expander.write(f"Zero Crossing Rate (Skewness): {zcr_skew:.4f}")
    stat_expander.write(f"Root Mean Square Energy (RMSE): {energy:.4f}")
    stat_expander.write(f"RMSE Median: {energy_median:.4f}")
    stat_expander.write(f"RMSE Standard Deviation: {energy_std_dev:.4f}")
    stat_expander.write(f"RMSE Kurtosis: {energy_kurtosis:.4f}")
    stat_expander.write(f"RMSE Skewness: {energy_skew:.4f}")
