import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .image_detector import ImageEmotionDetector
from .questionnaire_detector import QuestionnaireEmotionDetector
from .data_loader import SpotifyDataLoader
from .recommender import SpotifyRecommender
from .rag_enhancer import RAGEnhancer
from .constants import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)

class EmosiFacade:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B", data_path=None, use_dummy=False):
        logger.info("Initializing EMOSI system...")
        logger.info("Loading image emotion detector...")
        self.image_detector = ImageEmotionDetector(model_name, use_dummy=use_dummy)
        logger.info("Loading questionnaire emotion detector...")
        self.questionnaire_detector = QuestionnaireEmotionDetector()
        logger.info("Loading Spotify data and recommender...")
        self.data_loader = SpotifyDataLoader(data_path=data_path)
        self.music_recommender = SpotifyRecommender(self.data_loader)
        logger.info("Initializing RAG enhancer...")
        self.rag_enhancer = RAGEnhancer(self.music_recommender)
        logger.info("EMOSI system ready!")
    
    def detect_image_emotion(self, image_path: str) -> Tuple[str, Dict[str, float], np.ndarray]:
        logger.info(f"Analyzing image: {image_path}")
        dominant_emotion, emotion_scores = self.image_detector.detect_emotion(image_path)
        emotion_vector = self.image_detector.get_emotion_vector(emotion_scores)
        return dominant_emotion, emotion_scores, emotion_vector
    
    def get_questionnaire(self) -> List[Dict[str, Any]]:
        return self.questionnaire_detector.get_questionnaire()
    
    def process_questionnaire(self, responses: List[str]) -> Tuple[str, Dict[str, float], np.ndarray]:
        logger.info("Processing questionnaire responses...")
        dominant_emotion, emotion_scores = self.questionnaire_detector.process_responses(responses)
        emotion_vector = self.questionnaire_detector.get_emotion_vector(emotion_scores)
        return dominant_emotion, emotion_scores, emotion_vector
    
    def recommend_by_emotion(self, emotion_vector: np.ndarray, num_recommendations: int = 5) -> List[Dict]:
        recommendations = self.music_recommender.recommend_by_emotion(emotion_vector, num_recommendations)
        enhanced_recommendations = self.rag_enhancer.enhance_recommendations(
            recommendations,
            {'emotion': EMOTION_CATEGORIES[np.argmax(emotion_vector)]}
        )
        return enhanced_recommendations
    
    def recommend_by_text(self, query_text: str, num_recommendations: int = 5, year_cutoff: int = 2010) -> List[Dict]:
        try:
            recommendations = self.music_recommender.recommend_by_text(
                query_text=query_text,
                num_recommendations=num_recommendations,
                year_cutoff=year_cutoff
            )
            if self.rag_enhancer:
                query_context = {
                    "query_type": "text",
                    "query_text": query_text,
                }
                recommendations = self.rag_enhancer.enhance_recommendations(
                    recommendations=recommendations,
                    query_context=query_context
                )
            return recommendations
        except Exception as e:
            logging.error(f"Error with text-based recommendation: {e}")
            return []
    
    def recommend_by_similar_tracks(self, track_ids: List[str], num_recommendations: int = 5) -> List[Dict]:
        recommendations = self.music_recommender.recommend_by_similar_tracks(track_ids, num_recommendations)
        enhanced_recommendations = self.rag_enhancer.enhance_recommendations(
            recommendations,
            {'track_ids': track_ids}
        )
        return enhanced_recommendations
    
    def get_track_info(self, track_id: str) -> Dict:
        return self.music_recommender.get_track_info(track_id)
    
    def run_image_based_recommendation(self, image_path: str, num_recommendations: int = 5) -> Tuple[str, Dict[str, float], List[Dict]]:
        dominant_emotion, emotion_scores, emotion_vector = self.detect_image_emotion(image_path)
        recommendations = self.recommend_by_emotion(emotion_vector, num_recommendations)
        return dominant_emotion, emotion_scores, recommendations
    
    def run_text_based_recommendation(self, responses: List[str] = None, query_text: str = None,
                                      num_recommendations: int = 5, year_cutoff: int = 2010) -> Tuple[Optional[str], Optional[Dict[str, float]], List[Dict]]:
        try:
            dominant_emotion = None
            emotion_scores = None
            if responses:
                dominant_emotion, emotion_scores, emotion_vector = self.process_questionnaire(responses)
                recommendations = self.recommend_by_emotion(emotion_vector, num_recommendations)
            elif query_text:
                logger.info(f"Running text query-based recommendation with year cutoff: {year_cutoff}")
                recommendations = self.recommend_by_text(query_text, num_recommendations, year_cutoff)
            else:
                raise ValueError("Either questionnaire responses or text query must be provided")
            return dominant_emotion, emotion_scores, recommendations
        except Exception as e:
            logging.error(f"Error in text-based recommendation flow: {e}")
            return None, None, []
    
    def run_combined_recommendation(self, image_path: str, responses: List[str],
                                    num_recommendations: int = 5) -> Tuple[Dict[str, Any], List[Dict], List[Dict]]:
        image_emotion, image_scores, image_vector = self.detect_image_emotion(image_path)
        image_recommendations = self.recommend_by_emotion(image_vector, num_recommendations)
        questionnaire_emotion, questionnaire_scores, questionnaire_vector = self.process_questionnaire(responses)
        questionnaire_recommendations = self.recommend_by_emotion(questionnaire_vector, num_recommendations)
        emotion_similarity = self._compare_emotion_vectors(image_vector, questionnaire_vector)
        context = {
            'image_emotion': image_emotion,
            'questionnaire_emotion': questionnaire_emotion,
            'emotion_similarity': float(emotion_similarity),
            'explanation': self._get_emotion_difference_explanation(image_emotion, questionnaire_emotion)
        }
        return context, image_recommendations, questionnaire_recommendations
    
    def _compare_emotion_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        if not isinstance(vector1, np.ndarray):
            vector1 = np.array(vector1)
        if not isinstance(vector2, np.ndarray):
            vector2 = np.array(vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 > 0:
            vector1 = vector1 / norm1
        if norm2 > 0:
            vector2 = vector2 / norm2
        dot_product = np.dot(vector1, vector2)
        similarity = (dot_product + 1) / 2
        return float(similarity)
    
    def _get_emotion_difference_explanation(self, image_emotion: str, questionnaire_emotion: str) -> str:
        if image_emotion == questionnaire_emotion:
            return f"Both methods detected a primarily {image_emotion} emotion, suggesting strong consistency between your visual content and stated preferences."
        explanations = {
            ("happy", "excited"): "Your image appears happy while your responses indicate excitement. These are closely related positive emotions, with excitement suggesting higher energy.",
            ("excited", "happy"): "Your image shows excitement while your responses indicate happiness. Both are positive emotions, but the image suggests higher energy than your questionnaire responses.",
            ("sad", "calm"): "Your image conveys sadness, but your responses suggest a calmer emotion. Perhaps the visual content is more melancholic than your current mood.",
            ("calm", "sad"): "Your image appears calm, while your responses indicate sadness. The visual content may be more serene than your expressed emotional state.",
            ("happy", "calm"): "Your image appears happy, but your responses suggest a calmer emotion. This might indicate a preference for more relaxed music despite positive visual content.",
            ("calm", "happy"): "Your image conveys calmness, while your responses indicate happiness. You might be feeling more upbeat than your visual content suggests.",
            ("angry", "excited"): "Your image appears to convey anger, while your responses indicate excitement. Both emotions are high-energy but differ in valence (positive vs. negative).",
            ("neutral", "happy"): "Your image appears neutral, but your responses suggest happiness. You may prefer more upbeat music than your visual content would typically suggest.",
        }
        default_explanation = f"Your image conveys a primarily {image_emotion} emotion, while your questionnaire responses suggest a {questionnaire_emotion} emotion. This difference highlights how visual content can evoke different feelings than your stated preferences."
        return explanations.get((image_emotion, questionnaire_emotion), default_explanation)

