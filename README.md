# 🎵 Image-to-Music Recommender

A multimodal machine learning system that recommends music based on the **emotional context of an image**.

---

## 🧠 Problem Statement

Music and visuals are deeply connected through emotion and context. This project aims to bridge the two by answering:

> *“Given an image, what kind of music best matches its mood?”*

The system extracts semantic meaning from an image and maps it to audio features of songs to generate relevant recommendations.

---

## 🚀 Approach

The system follows a **multimodal pipeline** combining computer vision and music feature analysis:

### 1. Image Understanding

* A pretrained **CLIP (Contrastive Language–Image Pretraining)** model is used.
* The image is compared against a set of **semantic mood prompts** (e.g., “romantic couple”, “calm nature”).
* Output: Top-2 probable moods with confidence scores.

---

### 2. Mood Representation

* Each detected mood is mapped to a **numerical feature vector** representing musical characteristics:

  * valence (positivity)
  * energy (intensity)
  * danceability
  * tempo
  * loudness

* A **weighted combination** of top moods is computed to form a target “music profile”.

---

### 3. Music Recommendation

* Songs are selected from a Spotify dataset based on:

  * similarity to the target feature vector (cosine similarity)
  * filtering by popularity and genre for realistic results

* Output: Top-N songs that best match the image’s emotional context.

---

## 🧩 System Architecture

```id="2ktzrd"
Image → CLIP → Mood Detection → Feature Mapping → Similarity Search → Song Recommendations
```

---

## 📁 Project Structure

```id="lq3g4c"
image-music-recommender/
│
├── data/
│   └── spotify-tracks-dataset.csv   # music dataset
│
├── mood_detector.py                # CLIP-based mood detection
├── recommender.py                  # feature-based music recommender
├── app.py                          # Streamlit UI
├── requirements.txt
```

---

## ⚙️ Key Components

### 🔹 `mood_detector.py`

* Loads CLIP model
* Encodes images and text prompts
* Computes similarity scores
* Returns top-2 moods

---

### 🔹 `recommender.py`

* Loads and preprocesses Spotify dataset
* Normalizes audio features
* Maps moods → feature vectors
* Uses cosine similarity for recommendation

---

### 🔹 `app.py`

* Streamlit interface
* Handles image upload
* Displays detected moods and recommended songs

---

## 📊 Dataset

This project uses the Spotify Tracks Dataset from Kaggle:

👉 [Spotify Tracks Dataset](https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset?utm_source=chatgpt.com)

* Contains thousands of songs across multiple genres
* Includes rich audio features such as:

  * danceability
  * energy
  * valence
  * tempo
  * loudness
* Designed for tasks like recommendation systems and music analysis ([Hugging Face][1])

---

## ⚡ How It Works (Summary)

1. Upload an image
2. Extract semantic meaning using CLIP
3. Convert meaning → musical feature space
4. Find similar songs using vector similarity
5. Return relevant music recommendations

---

## 🔧 Setup Instructions

### 1. Clone repository

```bash id="3g6p1h"
git clone https://github.com/YOUR_USERNAME/image-music-recommender.git
cd image-music-recommender
```

### 2. Create virtual environment

```bash id="k0xht3"
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash id="u3cztf"
pip install -r requirements.txt
```

### 4. Run the app

```bash id="q5jyyx"
streamlit run app.py
```

---

## ⚠️ Notes

* First run may be slow due to model loading (~300MB CLIP model)
* GPU (CUDA) significantly improves performance
* Recommendations depend on dataset filtering and feature mapping

---

## 📌 Future Improvements

* Caption-based understanding (image → text → emotion)
* Integration with Spotify API for playback
* Personalized recommendations based on user history
* Improved mood taxonomy and embeddings

---

## 🙏 Acknowledgements

* CLIP model by OpenAI
* Spotify dataset contributors on Kaggle
* Dataset by Yash Dev (Kaggle)

---

[1]: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset?utm_source=chatgpt.com "maharshipandya/spotify-tracks-dataset"
