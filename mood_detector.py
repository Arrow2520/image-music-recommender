import torch
import clip
from PIL import Image

class MoodDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.moods = [
            "happy vibe",
            "sad emotional scene",
            "romantic couple",
            "party celebration",
            "calm peaceful nature",
            "dark gloomy scene",
            "energetic action"
        ]

        self.text_tokens = clip.tokenize(self.moods).to(self.device)

    def predict(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text_tokens)

            similarity = (image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()

        top_indices = probs.argsort()[-2:][::-1]

        results = [(self.moods[i], float(probs[i])) for i in top_indices]

        return results
