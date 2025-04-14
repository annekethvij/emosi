import numpy as np
import logging
import random
from typing import Dict, Tuple
from PIL import Image
import re
from .constants import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)

class ImageEmotionDetector:

    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B", use_dummy=False):
        logger.info(f"Initializing ImageEmotionDetector with model: {model_name}")
        self.model_name = model_name
        self.use_dummy = use_dummy
        if not use_dummy:
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
                from qwen_omni_utils import process_mm_info
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
                self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
                logger.info("Model and processor loaded successfully")
            except Exception as e:
                logger.error(f"Error initializing model: {e}")
                logger.warning("Falling back to dummy detection")
                self.use_dummy = True

    def detect_emotion(self, image_path: str) -> Tuple[str, Dict[str, float]]:
        logger.info(f"Detecting emotion from image: {image_path}")
        if self.use_dummy:
            print("\n=== DUMMY IMAGE ANALYSIS (No Model) ===")
            emotion_scores = {emotion: 0.1 for emotion in EMOTION_CATEGORIES}
            chosen_emotion = random.choice(EMOTION_CATEGORIES)
            emotion_scores[chosen_emotion] = 0.6
            total = sum(emotion_scores.values())
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            dominant_emotion = chosen_emotion
            print(f"Detected dominant emotion: {dominant_emotion}")
            print("Emotion scores:")
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.3f}")
            print("=== END DUMMY IMAGE ANALYSIS ===\n")
            return dominant_emotion, emotion_scores
        try:
            image = Image.open(image_path).convert("RGB")
            conversation = [
                {
                    "role": "system",
                    "content": "You are an emotion detection AI. First, describe the image in 2-3 sentences. Then analyze what emotion it primarily conveys. Only respond with one of these emotions: happy, sad, calm, excited, angry, relaxed, fearful, surprised, or neutral. Also provide a confidence score from 0 to 1 for each of these emotions, with the scores summing to 1."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            images = [image]
            inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.model.device)
            text_ids = self.model.generate(**inputs, max_new_tokens=256, return_audio=False)
            response = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0]
            print("\n=== IMAGE ANALYSIS ===")
            print("Full model response:")
            print(response)
            emotion_scores = self._parse_emotion_response(response)
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            print(f"Detected dominant emotion: {dominant_emotion}")
            print("Emotion scores:")
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.3f}")
            print("=== END IMAGE ANALYSIS ===\n")
            logger.info(f"Detected dominant emotion: {dominant_emotion}")
            return dominant_emotion, emotion_scores
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            default_scores = {emotion: 1.0/len(EMOTION_CATEGORIES) if emotion == "neutral" else 0.0 for emotion in EMOTION_CATEGORIES}
            default_scores["neutral"] = 1.0
            return "neutral", default_scores

    def _parse_emotion_response(self, response: str) -> Dict[str, float]:
        emotion_scores = {emotion: 0.01 for emotion in EMOTION_CATEGORIES}
        try:
            for emotion in EMOTION_CATEGORIES:
                if emotion in response.lower():
                    for pattern in [f"{emotion}: ([0-9.]+)", f"{emotion} \(([0-9.]+)\)"]:
                        match = re.search(pattern, response.lower())
                        if match:
                            try:
                                emotion_scores[emotion] = float(match.group(1))
                            except ValueError:
                                pass
            if max(emotion_scores.values()) <= 0.01:
                for emotion in EMOTION_CATEGORIES:
                    if emotion in response.lower():
                        emotion_scores[emotion] = 0.8
                        break
            if max(emotion_scores.values()) <= 0.01:
                indices = {}
                for emotion in EMOTION_CATEGORIES:
                    idx = response.lower().find(emotion)
                    if idx >= 0:
                        indices[emotion] = idx
                if indices:
                    first_emotion = min(indices.items(), key=lambda x: x[1])[0]
                    emotion_scores[first_emotion] = 0.8
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            return emotion_scores
        except Exception as e:
            logger.error(f"Error parsing emotion response: {e}")
            return {emotion: 1.0/len(EMOTION_CATEGORIES) for emotion in EMOTION_CATEGORIES}

    def get_emotion_vector(self, emotion_scores: Dict[str, float]) -> np.ndarray:
        """
        Convert emotion scores to a vector representation for FAISS.
        """
        vector = np.array([emotion_scores.get(emotion, 0.0) for emotion in EMOTION_CATEGORIES])
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

