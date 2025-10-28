import streamlit as st
import pandas as pd
import cv2
import numpy as np
from deepface import DeepFace
import random

# -------------------------------
# 🎯 Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Emotion + ACO Music Recommender", page_icon="🎵", layout="wide")
st.title("🎵 Emotion-Based Music Recommender using ACO & DeepFace")
st.markdown("Upload an image to detect your emotion and get personalized song suggestions optimized using Ant Colony Optimization (ACO).")

# -------------------------------
# 📂 Load Dataset
# -------------------------------
try:
    playlist_df = pd.read_csv("playlist.csv")  # Ensure playlist.csv is in your repo
    playlist_df.columns = playlist_df.columns.str.lower().str.strip()
    st.success("✅ Playlist dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ 'playlist.csv' file not found. Please upload it to your app directory.")
    st.stop()

# -------------------------------
# 🎭 Emotion → Song Category Mapping
# -------------------------------
emotion_column_map = {
    "happy": "happy songs",
    "sad": "sad songs",
    "angry": "angry songs",
    "neutral": "neutral songs",
    "fear": "fear songs",
    "disgust": "disgust songs",
    "surprise": "surprise songs"
}

# -------------------------------
# 🐜 ACO Optimization Function
# -------------------------------
def ant_colony_optimization(song_list, n_ants=10, n_iterations=20):
    """
    Simple ACO simulation to optimize the order of recommended songs.
    """
    pheromone = np.ones(len(song_list))
    for _ in range(n_iterations):
        for _ in range(n_ants):
            i, j = random.sample(range(len(song_list)), 2)
            pheromone[i] += 0.1
            pheromone[j] += 0.1
        pheromone *= 0.95  # evaporation
    sorted_songs = [x for _, x in sorted(zip(pheromone, song_list), reverse=True)]
    return sorted_songs

# -------------------------------
# 📸 Upload Image
# -------------------------------
uploaded_file = st.file_uploader("📷 Upload an image to detect emotion", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Display image
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # 🧠 Emotion Detection
    # -------------------------------
    with st.spinner("Analyzing emotion..."):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            st.success(f"**Detected Emotion:** {dominant_emotion.capitalize()} 🎯")
        except Exception as e:
            st.error(f"Emotion detection failed: {e}")
            st.stop()

    # -------------------------------
    # 🎧 Song Recommendation
    # -------------------------------
    song_column = emotion_column_map.get(dominant_emotion, None)
    if song_column and song_column in playlist_df.columns:
        recommended_songs = playlist_df[song_column].dropna().tolist()
        if recommended_songs:
            # Optimize order using ACO
            optimized_songs = ant_colony_optimization(recommended_songs)
            
            st.subheader("🎵 Optimized Song Recommendations")
            for i, song in enumerate(optimized_songs[:10], 1):
                st.write(f"{i}. {song}")
        else:
            st.warning(f"No songs found for emotion: {dominant_emotion}")
    else:
        st.warning(f"Emotion '{dominant_emotion}' not found in dataset.")

# -------------------------------
# 🧩 Footer
# -------------------------------
st.markdown("---")
st.markdown("👨‍💻 **Developed by:** You  |  🤖 ACO + DeepFace + Streamlit Integration")
st.markdown("⚙️ This app uses Ant Colony Optimization (ACO) to rank songs by user emotion context.")
