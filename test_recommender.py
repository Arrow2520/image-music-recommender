from mood_detector import MoodDetector
from recommender import MusicRecommender

detector = MoodDetector()
recommender = MusicRecommender("data/spotify-tracks-dataset.csv")

image_path = "test.jpeg"

# Step 1: Detect mood
moods = detector.predict(image_path)

print("Detected moods:")
for mood, score in moods:
    print(f"{mood} → {score:.3f}")

# Step 2: Recommend songs
songs = recommender.recommend(moods)

print("\nRecommended Songs:")
print(songs.to_string(index=False))
