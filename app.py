# ============================================
# 🎵 Streamlit App: Emotion-Based ACO Playlist
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import random
from deepface import DeepFace
from PIL import Image
import tempfile

# ------------------------------
# Load Dataset
# ------------------------------
st.title("🎭 Emotion-Based Music Recommender using ACO Optimization")

uploaded_file = st.file_uploader("📂 Upload your playlist.csv file", type=["csv"])
if uploaded_file:
    playlist_df = pd.read_csv(uploaded_file)
    playlist_df.columns = playlist_df.columns.str.lower().str.strip()
    st.success("✅ Playlist loaded successfully!")
else:
    st.warning("⚠️ Please upload a playlist.csv file to continue.")
    st.stop()

# Emotion mapping
emotion_column_map = {
    "happy": "happy songs",
    "sad": "sad songs",
    "angry": "angry songs",
    "neutral": "neutral songs",
    "fear": "fear songs",
    "disgust": "disgust songs",
    "surprise": "surprise songs"
}

# ------------------------------
# ACO Optimization Function
# ------------------------------
def calculate_transition_cost(song1, song2):
    return abs(hash(song1) - hash(song2)) % 100

def aco_optimize_playlist(songs, num_ants=10, num_iterations=30, alpha=1, beta=2, rho=0.5):
    n = len(songs)
    if n <= 1:
        return songs
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i][j] = calculate_transition_cost(songs[i], songs[j])
    pheromone = np.ones((n, n))
    best_path, best_cost = None, float('inf')
    for _ in range(num_iterations):
        all_paths = []
        for _ in range(num_ants):
            path = [random.randint(0, n - 1)]
            while len(path) < n:
                i = path[-1]
                probs = []
                for j in range(n):
                    if j not in path:
                        probs.append((pheromone[i][j] ** alpha) * ((1.0 / (cost_matrix[i][j] + 1e-9)) ** beta))
                    else:
                        probs.append(0)
                probs = np.array(probs)
                probs /= probs.sum() if probs.sum() != 0 else 1
                next_node = np.random.choice(range(n), p=probs)
                path.append(next_node)
            total_cost = sum(cost_matrix[path[k]][path[k + 1]] for k in range(n - 1))
            all_paths.append((path, total_cost))
            if total_cost < best_cost:
                best_path, best_cost = path, total_cost
        pheromone *= (1 - rho)
        for path, cost in all_paths:
            for k in range(len(path) - 1):
                pheromone[path[k]][path[k + 1]] += 1.0 / cost
    return [songs[i] for i in best_path]

# ------------------------------
# Webcam Image Input
# ------------------------------
st.subheader("📸 Capture or Upload an Image to Detect Emotion")
img_file = st.camera_input("Take a photo") or st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

if img_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(img_file.read())
        image_path = temp_file.name

    # Analyze emotion using DeepFace
    with st.spinner("Analyzing emotion..."):
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        st.success(f"🧠 Detected Emotion: **{dominant_emotion.capitalize()}**")

    # ------------------------------
    # Fetch and Optimize Playlist
    # ------------------------------
    song_column = emotion_column_map.get(dominant_emotion, None)
    if song_column and song_column in playlist_df.columns:
        recommended_songs = playlist_df[song_column].dropna().tolist()
        if recommended_songs:
            optimized_songs = aco_optimize_playlist(recommended_songs[:10])
            st.subheader("🎶 Optimized Playlist")
            for idx, song in enumerate(optimized_songs, 1):
                st.write(f"{idx}. {song}")
        else:
            st.warning("No songs found for this emotion.")
    else:
        st.error(f"Emotion '{dominant_emotion}' not found in dataset columns.")
else:
    st.info("👆 Please capture or upload an image to start emotion detection.")
