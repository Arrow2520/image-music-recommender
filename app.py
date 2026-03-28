import streamlit as st
from mood_detector import MoodDetector
from recommender import MusicRecommender
from PIL import Image

# Load models
detector = MoodDetector()
recommender = MusicRecommender("data/spotify-tracks-dataset.csv")

st.title("🎵 Image to Music Recommender")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp image
    image_path = "temp.jpg"
    image.save(image_path)

    # Predict mood
    moods = detector.predict(image_path)

    st.subheader("Detected Moods:")
    for mood, score in moods:
        st.write(f"{mood} → {score:.3f}")

    # Recommend songs
    songs = recommender.recommend(moods)

    st.subheader("🎶 Recommended Songs:")
    st.dataframe(songs)
