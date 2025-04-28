import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎧 Prediksi & Rekomendasi Genre Musik", layout="centered")
st.title("🎧 Prediksi Genre & Rekomendasi Lagu Favorit")

# 1️⃣ Load model, scaler, encoder, dan data
model = load_model("fcnn_genre_classifier90.h5", compile=False)
scaler = joblib.load("scaler90.pkl")
encoder = joblib.load("encoder90.pkl")
data = joblib.load("7000SONGS_NEW2.pkl")

# 2️⃣ Validasi data
if not isinstance(data, pd.DataFrame):
    st.error("File data bukan DataFrame. Pastikan formatnya benar.")
    st.stop()

# 3️⃣ Pilih 3 lagu dari dropdown
st.subheader("🎵 Pilih 3 Lagu")
all_filenames = data["filename"].tolist()

lagu1 = st.selectbox("Lagu ke-1", all_filenames, key="lagu1")
lagu2 = st.selectbox("Lagu ke-2", [f for f in all_filenames if f != lagu1], key="lagu2")
lagu3 = st.selectbox("Lagu ke-3", [f for f in all_filenames if f != lagu1 and f != lagu2], key="lagu3")

selected_filenames = [lagu1, lagu2, lagu3]

# 🔁 Fungsi rekomendasi berdasarkan KNN
def get_song_recommendations(selected_songs, song_data, max_recommendations=15):
    feature_columns = song_data.columns[2:]  # skip 'filename' dan 'genre'
    song_features = song_data[feature_columns].values
    all_recommended_songs = []

    for song in selected_songs:
        song_index = song_data[song_data['filename'] == song].index[0]
        song_feature = song_features[song_index].reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=max_recommendations, metric='cosine')
        knn.fit(song_features)

        distances, indices = knn.kneighbors(song_feature, n_neighbors=max_recommendations)

        for i in indices[0]:
            song_name = song_data['filename'][i]
            if song_name not in selected_songs and song_name not in all_recommended_songs:
                all_recommended_songs.append(song_name)

        if len(all_recommended_songs) >= max_recommendations:
            break

    if len(all_recommended_songs) < max_recommendations:
        remaining = max_recommendations - len(all_recommended_songs)
        for i in range(len(song_data)):
            song_name = song_data['filename'][i]
            if song_name not in selected_songs and song_name not in all_recommended_songs:
                all_recommended_songs.append(song_name)
                if len(all_recommended_songs) == max_recommendations:
                    break

    return all_recommended_songs[:max_recommendations]

# Add slider to control the number of recommendations
num_recommendations = st.slider(
    "🔢 Jumlah Rekomendasi Lagu",
    min_value=0,  # minimum 0 recommendations
    max_value=30,  # maximum 30 recommendations
    value=15,  # default value (this can be adjusted to your preference)
    step=5  # step size for each increment
)

if st.button("🔍 Prediksi Genre & Rekomendasi"):
    sample = data[data["filename"].isin(selected_filenames)].reset_index(drop=True)
    features = sample.drop(columns=["filename", "genre"])
    features_numeric = features.select_dtypes(include=[np.number])
    X_scaled = scaler.transform(features_numeric)

    pred_probs = model.predict(X_scaled)
    pred_indices = np.argmax(pred_probs, axis=1)
    pred_labels = encoder.inverse_transform(pred_indices)

    st.subheader("📈 Hasil Prediksi Genre per Lagu")
    hasil = pd.DataFrame({
        "Filename": sample["filename"],
        "Actual Genre": sample["genre"],
        "Predicted Genre": pred_labels
    })
    st.dataframe(hasil)

    # 🎯 Tambahkan Tabel Probabilitas Setiap Genre
    probs_df = pd.DataFrame(pred_probs, columns=encoder.classes_)
    probs_df.insert(0, "Filename", sample["filename"])

    # 🔝 Prediksi genre favorit dari total probabilitas
    total_probs = np.sum(pred_probs, axis=0)
    genre_index_favorit = np.argmax(total_probs)
    genre_favorit = encoder.inverse_transform([genre_index_favorit])[0]

    st.subheader("🎵 Genre Favorit Kamu (Versi Model)")
    st.success(f"✅ Berdasarkan 3 lagu yang kamu pilih, kamu kemungkinan suka genre **{genre_favorit.upper()}** 🎶")

    # 🔁 Rekomendasi Lagu dari KNN
    st.subheader(f"🎶 Rekomendasi Lagu Serupa (Berdasarkan KNN) - {num_recommendations} Lagu")
    rekomendasi_knn = get_song_recommendations(selected_filenames, data, max_recommendations=num_recommendations)
    rekomendasi_df = data[data["filename"].isin(rekomendasi_knn)][["filename", "genre"]].reset_index(drop=True)
    st.dataframe(rekomendasi_df)
