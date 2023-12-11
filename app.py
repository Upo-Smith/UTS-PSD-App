import streamlit as st
import numpy as np
import scipy as sc
import pandas as pd
import librosa
import pickle

st.title("Prediksi Emosi Berdasarkan Audio Menggunakan KNN")

audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

st.markdown("**DISCLAIMER**: mungkin nanti ada error, tapi itu karena belum ada audio yg dimasukkan, setelah audio dimasukkan error akan menghilang")

tab1, tab2, tab3 = st.tabs(["Extract", "Preprocessing", "Prediction"])

norm = []
zero = []
root = []
data = []

if audio is not None:
    x, sr = librosa.load(audio, sr=None)

    freqs = np.fft.fftfreq(x.size)

    mean = np.mean(freqs)
    median = np.median(freqs)
    std = np.std(freqs)
    minv = np.amin(freqs) 
    maxv = np.amax(freqs)
    mode = sc.stats.mode(freqs)[0]
    skew = sc.stats.skew(freqs)
    kurt = sc.stats.kurtosis(freqs)
    q1 = np.quantile(freqs, 0.25)
    q3 = np.quantile(freqs, 0.75)
    iqr = sc.stats.iqr(freqs)

    norm = pd.DataFrame({
        'Mean':[mean],
        'Median':[median],
        'Modus':[mode],
        'Standar Deviasi':[std],
        'Nilai Min':[minv],
        'Nilai Max':[maxv],
        'Skewness':[skew],
        'Kurtosis':[kurt],
        'Q1':[q1],
        'Q3':[q3],
        'Iqr':[iqr]
    })
    
    # Zero Crossing Rate
    y = librosa.feature.zero_crossing_rate(x)
    zcr = []
    for i in y:
        for j in i:
            zcr.append(j)
    mean_zcr = np.mean(zcr) 
    median_zcr = np.median(zcr)
    std_zcr = np.std(zcr) 
    skew_zcr = sc.stats.skew(zcr)
    kurt_zcr = sc.stats.kurtosis(zcr)

    zero = pd.DataFrame({
        'ZCR_mean':[mean_zcr],
        'ZCR_median':[median_zcr],
        'ZCR_std':[std_zcr],
        'ZCR_skew':[skew_zcr],
        'ZCR_kurt':[kurt_zcr]
    })

    # Root Mean Square Energy
    z = librosa.feature.rms(y=x)
    rms = []
    for i in z:
        for j in i:
            rms.append(j)
    mean_rms = np.mean(rms) 
    median_rms = np.median(rms) 
    std_rms = np.std(rms) 
    skew_rms = sc.stats.skew(rms) 
    kurt_rms = sc.stats.kurtosis(rms) 

    root = pd.DataFrame({
        'RMS_mean':[mean_rms],
        'RMS_median':[median_rms],
        'RMS_std':[std_rms],
        'RMS_skew':[skew_rms],
        'RMS_kurt':[kurt_rms]
    })
    
    data = pd.DataFrame({
        'mean':[mean],
        'median':[median],
        'modus':[mode],
        'std':[std],
        'min':[minv],
        'max':[maxv],
        'skewness':[skew],
        'kurtosis':[kurt],
        'q1':[q1],
        'q3':[q3],
        'iqr':[iqr],
        'zcr_mean':[mean_zcr],
        'zcr_median':[median_zcr],
        'zcr_std':[std_zcr],
        'zcr_skew':[skew_zcr],
        'zcr_kurt':[kurt_zcr],
        'rmse_mean':[mean_rms],
        'rmse_median':[median_rms],
        'rmse_std':[std_rms],
        'rmse_skew':[skew_rms],
        'rmse_kurt':[kurt_rms]
    })
    
with tab1:
    st.write("Ekstraksi Data")
    st.dataframe(norm)
    st.write("Zero Crossing Rate")
    st.dataframe(zero)
    st.write("Root Mean Square Energy")
    st.dataframe(root)

with tab2:
    method = st.write('Normalisasi hasil ekstraksi inputan menggunakan Z-Score Tanpa PCA:')

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))

    data_sc = scaler.transform(data)
    data_pca = pca.transform(data_sc)

    st.write(data_sc)

with tab3:
    Pclf = pickle.load(open('clf.pkl', 'rb'))
    Sclf = pickle.load(open('Sclf.pkl', 'rb'))

    st.write("Hasil prediksi dari audio di atas (Akurasi 81%, K = 9):")

    predict1 = Sclf.predict(data_sc)
    st.write(predict1)
