import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class MusicRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Select required columns
        self.df = self.df[[
            "track_name",
            "artists",
            "valence",
            "energy",
            "danceability",
            "tempo",
            "loudness",
            "popularity",
            "track_genre"
        ]].dropna()

        # 🔥 Filter 1: Popular songs only
        self.df = self.df[self.df["popularity"] > 50]

        # 🔥 Filter 2 (optional but recommended): restrict genres
        self.df = self.df[self.df["track_genre"].isin([
            "pop", "acoustic", "indie", "romance", "chill", "sad"
        ])]

        # Feature matrix
        self.features = self.df[[
            "valence",
            "energy",
            "danceability",
            "tempo",
            "loudness"
        ]].values

        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        # 🔥 Improved mood mapping (5D)
        self.mood_map = {
            "happy vibe": [0.8, 0.6, 0.7, 120, -5],
            "sad emotional scene": [0.2, 0.3, 0.2, 70, -15],
            "romantic couple": [0.7, 0.3, 0.6, 80, -12],
            "party celebration": [0.9, 0.9, 0.8, 130, -3],
            "calm peaceful nature": [0.5, 0.2, 0.3, 60, -20],
            "dark gloomy scene": [0.2, 0.2, 0.1, 65, -18],
            "energetic action": [0.7, 0.9, 0.8, 140, -4]
        }

    def recommend(self, moods, n=5):
        vectors = []
        weights = []

        for mood, score in moods:
            if mood in self.mood_map:
                vectors.append(self.mood_map[mood])
                weights.append(score)

        if not vectors:
            return []

        # Weighted average mood vector
        target = np.average(vectors, axis=0, weights=weights)

        # Normalize same as dataset
        target = self.scaler.transform([target])[0]

        # Compute similarity
        similarities = cosine_similarity([target], self.features)[0]

        # 🔥 Take top 20 then randomly pick n (more natural results)
        top_indices = similarities.argsort()[-20:]

        if len(top_indices) < n:
            selected = top_indices
        else:
            selected = np.random.choice(top_indices, size=n, replace=False)

        return self.df.iloc[selected][["track_name", "artists", "track_genre"]]
