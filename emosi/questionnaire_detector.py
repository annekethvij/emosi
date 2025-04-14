import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from .constants import EMOTION_CATEGORIES, QUESTIONNAIRE

logger = logging.getLogger(__name__)

class QuestionnaireEmotionDetector:
    """
    To Detects emotions based on user responses to a questionnaire. Still need to add this to the main calling TODO for now!
    """
    def __init__(self):
        """Initialize the questionnaire emotion detector."""
        self.questions = QUESTIONNAIRE
        self.emotion_categories = EMOTION_CATEGORIES
        self.response_emotion_mapping = self._create_response_emotion_mapping()

    def _create_response_emotion_mapping(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Create a mapping from question responses to emotion weights.
        
        Returns:
            A nested dictionary mapping question indices, response options,
            and emotions to confidence scores.
        """
        mapping = {
            "0": {
                "Happy and cheerful": {"happy": 0.9, "excited": 0.6, "relaxed": 0.2},
                "Sad or melancholic": {"sad": 0.9, "fearful": 0.2, "calm": 0.1},
                "Calm and peaceful": {"calm": 0.9, "relaxed": 0.7, "neutral": 0.3},
                "Excited or energetic": {"excited": 0.9, "happy": 0.6, "surprised": 0.3},
                "Angry or frustrated": {"angry": 0.9, "fearful": 0.3, "sad": 0.2},
                "Relaxed": {"relaxed": 0.9, "calm": 0.7, "neutral": 0.2},
                "Fearful or anxious": {"fearful": 0.9, "sad": 0.4, "surprised": 0.2},
                "Surprised": {"surprised": 0.9, "excited": 0.3, "fearful": 0.2},
                "Neutral": {"neutral": 0.9, "calm": 0.3, "relaxed": 0.2}
            },
            "1": {
                "High energy": {"excited": 0.8, "happy": 0.6, "angry": 0.4},
                "Medium energy": {"neutral": 0.7, "happy": 0.5, "calm": 0.3},
                "Low energy/relaxing": {"relaxed": 0.8, "calm": 0.7, "sad": 0.3}
            },
            "2": {
                "Celebrating a happy moment": {"happy": 0.8, "excited": 0.7, "surprised": 0.3},
                "Sharing a meaningful/serious event": {"calm": 0.6, "neutral": 0.5, "sad": 0.4},
                "Just a casual update": {"neutral": 0.8, "relaxed": 0.5, "happy": 0.3},
                "Travel or adventure": {"excited": 0.8, "happy": 0.6, "surprised": 0.4},
                "Reflective/thoughtful moment": {"calm": 0.7, "relaxed": 0.5, "sad": 0.3}
            }
        }
        return mapping

    def get_questionnaire(self) -> List[Dict[str, Any]]:
        return self.questions

    def process_responses(self, responses: List[str]) -> Tuple[str, Dict[str, float]]:
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_categories}
        for q_idx, response in enumerate(responses):
            q_idx_str = str(q_idx)
            if q_idx_str not in self.response_emotion_mapping or response not in self.response_emotion_mapping[q_idx_str]:
                continue
            response_mapping = self.response_emotion_mapping[q_idx_str][response]
            for emotion, score in response_mapping.items():
                emotion_scores[emotion] += score
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Questionnaire detected emotion: {dominant_emotion}")
        logger.debug(f"Emotion scores: {emotion_scores}")
        return dominant_emotion, emotion_scores

    def get_emotion_vector(self, emotion_scores: Dict[str, float]) -> np.ndarray:
        vector = np.array([emotion_scores.get(emotion, 0.0) for emotion in self.emotion_categories])
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

