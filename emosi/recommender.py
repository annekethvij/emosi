import numpy as np
import logging
import faiss
import pandas as pd
from typing import Dict, List, Tuple, Any
from .constants import EMOTION_CATEGORIES
from .data_loader import SpotifyDataLoader

logger = logging.getLogger(__name__)

class SpotifyRecommender:
    def __init__(self, data_loader: SpotifyDataLoader):
        self.data_loader = data_loader
        self.tracks_df = data_loader.get_tracks()
        self.track_emotion_vectors = data_loader.get_track_emotion_vectors()
        self.track_embeddings = data_loader.get_track_embeddings()
        self.emotion_index = self._create_emotion_index()
        self.embedding_index = self._create_embedding_index()

    def _create_emotion_index(self) -> faiss.Index:
        try:
            logger.info("Creating FAISS index for track emotion vectors...")
            vectors = self.track_emotion_vectors.astype(np.float32)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            logger.info(f"FAISS emotion index created with {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS emotion index: {e}")
            raise

    def _create_embedding_index(self) -> faiss.Index:
        try:
            logger.info("Creating FAISS index for track embeddings...")
            vectors = self.track_embeddings.astype(np.float32)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            logger.info(f"FAISS embedding index created with {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS embedding index: {e}")
            raise

    def recommend_by_emotion(self, emotion_vector: np.ndarray, num_recommendations: int = 5) -> List[Dict]:
        try:
            if not isinstance(emotion_vector, np.ndarray):
                emotion_vector = np.array(emotion_vector)
            emotion_vector = emotion_vector.reshape(1, -1).astype(np.float32)
            print("\n=== EMOTION-BASED RECOMMENDATION SEARCH ===")
            print("Emotion vector used for search:")
            print(emotion_vector)
            print(f"Vector shape: {emotion_vector.shape}")
            dominant_emotion_idx = np.argmax(emotion_vector[0])
            dominant_emotion = EMOTION_CATEGORIES[dominant_emotion_idx]
            print(f"Dominant emotion in vector: {dominant_emotion} (index {dominant_emotion_idx})")
            distances, indices = self.emotion_index.search(emotion_vector, num_recommendations)
            print(f"FAISS search results - indices: {indices[0]}, distances: {distances[0]}")
            recommendations = []
            tracks_df = self.data_loader.get_tracks()
            for idx in indices[0]:
                if idx < len(tracks_df):
                    track = tracks_df.iloc[idx]
                    recommendation = {
                        'track_id': str(idx),
                        'track_name': track['name'] if 'name' in track else track['artists'],
                        'artist': track['artists'],
                        'album_name': track.get('album_name', 'Unknown'),
                        'genre': track.get('track_genre', 'Unknown'),
                        'duration_ms': int(track.get('duration_ms', 0)),
                        'popularity': int(track.get('popularity', 0)),
                        'distance': float(distances[0][list(indices[0]).index(idx)]),
                        'similarity': float(1.0 - distances[0][list(indices[0]).index(idx)] / (distances[0].max() + 1e-5)),
                        'track_idx': int(track['track_idx'])
                    }
                    recommendations.append(recommendation)
            print("=== END EMOTION-BASED RECOMMENDATION SEARCH ===\n")
            return recommendations
        except Exception as e:
            logger.error(f"Error recommending by emotion: {e}")
            print(f"Exception during recommendation: {e}")
            return []
    
    # Handle slider-based emotion inputs
    def recommend_by_emotion_sliders(self, emotion_sliders: Dict[str, float], num_recommendations: int = 5) -> List[Dict]:
        """
        Generate recommendations based on emotion slider values.
        
        Args:
            emotion_sliders: Dictionary mapping emotion names to intensity values (0-10)
            num_recommendations: Number of tracks to recommend
            
        Returns:
            List of track recommendations
        """
        try:
            logger.info("Processing emotion slider inputs for recommendations")
            
            # Filter out emotions with zero intensity
            active_emotions = {k: v for k, v in emotion_sliders.items() if v > 0}
            
            if not active_emotions:
                logger.warning("No active emotions found in slider input")
                # Return empty list or fallback to neutral recommendations
                return self.recommend_by_emotion(np.zeros(len(EMOTION_CATEGORIES)), num_recommendations)
            
            # Create a weighted emotion vector based on slider values
            emotion_vector = np.zeros(len(EMOTION_CATEGORIES))
            
            for emotion, intensity in active_emotions.items():
                # Normalize intensity to 0-1 range
                normalized_intensity = intensity / 10.0
                
                # Find the emotion in our categories
                for i, category in enumerate(EMOTION_CATEGORIES):
                    # Perform case-insensitive matching and handle compound emotions
                    if category.lower() in emotion.lower() or emotion.lower() in category.lower():
                        emotion_vector[i] = normalized_intensity
                        logger.info(f"Mapped emotion '{emotion}' to category '{category}' with intensity {normalized_intensity}")
            
            # Normalize the vector if it contains any non-zero values
            if np.sum(emotion_vector) > 0:
                emotion_vector = emotion_vector / np.sum(emotion_vector)
            
            logger.info(f"Created weighted emotion vector: {emotion_vector}")
            
            # Get recommendations using the existing method
            return self.recommend_by_emotion(emotion_vector, num_recommendations)
            
        except Exception as e:
            logger.error(f"Error processing emotion sliders: {e}", exc_info=True)
            return []
    
    # New function to incorporate image analysis with emotion recommendations
    def recommend_by_image_and_sliders(self, emotion_sliders: Dict[str, float], 
                                       image_features: Dict[str, float] = None, 
                                       num_recommendations: int = 5) -> List[Dict]:
        """
        Generate recommendations based on both emotion sliders and image analysis.
        
        Args:
            emotion_sliders: Dictionary mapping emotion names to intensity values (0-10)
            image_features: Optional dictionary of extracted image features
            num_recommendations: Number of tracks to recommend
            
        Returns:
            List of track recommendations with enhanced context
        """
        try:
            logger.info("Processing combined image and emotion slider inputs")
            
            # Start with emotion slider processing
            emotion_vector = np.zeros(len(EMOTION_CATEGORIES))
            active_emotions = {k: v for k, v in emotion_sliders.items() if v > 0}
            
            for emotion, intensity in active_emotions.items():
                normalized_intensity = intensity / 10.0
                for i, category in enumerate(EMOTION_CATEGORIES):
                    if category.lower() in emotion.lower() or emotion.lower() in category.lower():
                        emotion_vector[i] = normalized_intensity
            
            # If we have image features, adjust the emotion vector
            if image_features:
                logger.info("Incorporating image features into recommendation")
                
                # Map visual brightness to valence
                if 'brightness' in image_features:
                    # Find index of happiness/joy and boost it based on brightness
                    for i, category in enumerate(EMOTION_CATEGORIES):
                        if category.lower() in ['happy', 'happiness', 'joy']:
                            emotion_vector[i] += image_features['brightness'] * 0.3
                
                # Map color temperature to emotional tone
                if 'color_temperature' in image_features:
                    temp = image_features['color_temperature']
                    # Warm colors boost positive emotions
                    if temp > 0.6:
                        for i, category in enumerate(EMOTION_CATEGORIES):
                            if category.lower() in ['happy', 'joy', 'excitement', 'love']:
                                emotion_vector[i] += 0.2
                    # Cool colors boost melancholic emotions
                    elif temp < 0.4:
                        for i, category in enumerate(EMOTION_CATEGORIES):
                            if category.lower() in ['sad', 'sadness', 'melancholic']:
                                emotion_vector[i] += 0.2
                
                # Map visual complexity to musical complexity
                if 'visual_complexity' in image_features:
                    complexity = image_features['visual_complexity']
                    # High complexity might suggest more intricate music
                    if complexity > 0.7:
                        for i, category in enumerate(EMOTION_CATEGORIES):
                            if category.lower() in ['chaotic', 'wild', 'excitement']:
                                emotion_vector[i] += 0.15
            
            # Normalize the vector again after adjustments
            if np.sum(emotion_vector) > 0:
                emotion_vector = emotion_vector / np.sum(emotion_vector)
            
            logger.info(f"Final adjusted emotion vector: {emotion_vector}")
            
            # Get recommendations using the emotion vector
            recommendations = self.recommend_by_emotion(emotion_vector, num_recommendations)
            
            # Add image context to recommendations
            if image_features and recommendations:
                for rec in recommendations:
                    rec['image_context'] = True
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing image and sliders: {e}", exc_info=True)
            return []

    def recommend_by_text(self, query_text: str, num_recommendations: int = 5, year_cutoff: int = 2010) -> List[Dict]:
        logging.info("Starting text-based recommendation process")
        logging.info(f"Will prioritize songs released after {year_cutoff}")
        tracks_df = self.data_loader.get_tracks()
        logging.info(f"DataFrame columns: {list(tracks_df.columns)}")
        try:
            query_embedding = self._compute_fallback_query_embedding(query_text)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            logging.info(f"Query embedding shape before FAISS search: {query_embedding.shape}")
            extra_candidates = max(100, num_recommendations * 10)
            distances, indices = self.embedding_index.search(query_embedding, extra_candidates)
            logging.info(f"Raw indices from FAISS (first 10 candidates): {indices[0][:10]}...")
            candidates = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < len(tracks_df):
                    track = tracks_df.iloc[idx]
                    track_name = track['name'] if 'name' in track else track['artists']
                    artist_name = track['artists']
                    year = None
                    for year_col in ['year', 'release_date', 'release_year']:
                        if year_col in track and pd.notna(track[year_col]):
                            try:
                                year_val = track[year_col]
                                if isinstance(year_val, str) and len(year_val) >= 4:
                                    year = int(year_val[:4])
                                else:
                                    year = int(year_val)
                                break
                            except (ValueError, TypeError):
                                continue
                    if year is None:
                        year = 0
                    recommendation = {
                        'track_id': str(idx),
                        'track_name': track_name,
                        'artist': artist_name,
                        'track_idx': int(track['track_idx']),
                        'year': year,
                        'distance': float(distances[0][i])
                    }
                    candidates.append(recommendation)
                    if i < 30:
                        logging.info(f"Candidate: {track_name} - Year: {year}")
                else:
                    logging.warning(f"Index {idx} is out of bounds for DataFrame with size {len(tracks_df)}")
            logging.info(f"Total collected candidates: {len(candidates)}")
            newer_songs = [c for c in candidates if c['year'] >= year_cutoff]
            older_songs = [c for c in candidates if c['year'] < year_cutoff]
            logging.info(f"Newer songs (>= {year_cutoff}): {len(newer_songs)}")
            logging.info(f"Older songs (< {year_cutoff}): {len(older_songs)}")
            newer_songs.sort(key=lambda x: x['distance'])
            older_songs.sort(key=lambda x: (-x['year'], x['distance']))
            if older_songs:
                logging.info(f"Top 5 older songs years (sorted by newest first): {[s['year'] for s in older_songs[:5]]}")
            recommendations = newer_songs[:num_recommendations]
            if len(recommendations) < num_recommendations:
                remaining_spots = num_recommendations - len(recommendations)
                recommendations.extend(older_songs[:remaining_spots])
            logging.info(f"Final recommendations count: {len(recommendations)}")
            logging.info(f"Years of recommended songs: {[r['year'] for r in recommendations]}")
            for r in recommendations:
                r['release_year'] = r['year']
            return recommendations
        except Exception as e:
            logging.error(f"Error recommending by text: {e}")
            logging.error("Exception details:", exc_info=True)
            raise

    def _compute_fallback_query_embedding(self, query_text: str) -> np.ndarray:
        query_text = query_text.lower()
        feature_vector = np.zeros(self.track_embeddings.shape[1])
        keyword_mappings = {
            'danceable': {'danceability': 0.9, 'energy': 0.7, 'valence': 0.8},
            'dance': {'danceability': 0.8, 'energy': 0.6},
            'energetic': {'energy': 0.9, 'danceability': 0.7},
            'powerful': {'energy': 0.8, 'loudness': 0.7},
            'intense': {'energy': 0.8, 'loudness': 0.8},
            'loud': {'energy': 0.7, 'loudness': 0.9},
            'happy': {'valence': 0.9, 'energy': 0.7},
            'positive': {'valence': 0.8, 'energy': 0.6},
            'upbeat': {'valence': 0.8, 'energy': 0.7, 'danceability': 0.6},
            'sad': {'valence': 0.2, 'energy': 0.4},
            'melancholic': {'valence': 0.3, 'energy': 0.4},
            'depressing': {'valence': 0.1, 'energy': 0.3},
            'acoustic': {'acousticness': 0.9, 'instrumentalness': 0.7},
            'instrumental': {'instrumentalness': 0.9, 'acousticness': 0.6},
            'vocal': {'instrumentalness': 0.1, 'speechiness': 0.6},
            'fast': {'tempo': 0.8, 'energy': 0.7},
            'slow': {'tempo': 0.2, 'energy': 0.4},
            'mellow': {'tempo': 0.3, 'energy': 0.3, 'acousticness': 0.6},
            'major': {'mode': 1.0, 'valence': 0.7},
            'minor': {'mode': 0.0, 'valence': 0.4},
            'rock': {'energy': 0.8, 'valence': 0.6},
            'pop': {'danceability': 0.7, 'valence': 0.7},
            'hip hop': {'danceability': 0.8, 'speechiness': 0.7},
            'rap': {'speechiness': 0.8, 'danceability': 0.7},
            'jazz': {'instrumentalness': 0.7, 'acousticness': 0.8},
            'classical': {'acousticness': 0.9, 'instrumentalness': 0.8},
            'edm': {'danceability': 0.9, 'energy': 0.9},
            'electronic': {'danceability': 0.8, 'energy': 0.8},
            'r&b': {'valence': 0.7, 'danceability': 0.7},
            'country': {'acousticness': 0.7, 'valence': 0.6},
            'folk': {'acousticness': 0.9, 'instrumentalness': 0.6},
            'peaceful': {'energy': 0.3, 'valence': 0.6, 'acousticness': 0.7},
            'relaxing': {'energy': 0.2, 'valence': 0.6, 'acousticness': 0.8},
            'chill': {'energy': 0.4, 'valence': 0.6, 'acousticness': 0.6},
            'angry': {'energy': 0.9, 'valence': 0.2},
            'aggressive': {'energy': 0.9, 'valence': 0.3, 'loudness': 0.8},
            'romantic': {'valence': 0.7, 'acousticness': 0.6},
            'dramatic': {'valence': 0.4, 'energy': 0.7, 'loudness': 0.7},
            'happiness': {'valence': 0.9, 'energy': 0.7, 'tempo': 0.8},
            'joy': {'valence': 0.9, 'energy': 0.7, 'tempo': 0.8},
            'sadness': {'valence': 0.2, 'energy': 0.3, 'tempo': 0.3},
            'fear': {'valence': 0.3, 'energy': 0.6, 'mode': 0.0},
            'fearful': {'valence': 0.3, 'energy': 0.6, 'mode': 0.0},
            'surprise': {'energy': 0.7, 'valence': 0.6},
            'surprised': {'energy': 0.7, 'valence': 0.6},
            'disgust': {'valence': 0.2, 'energy': 0.5, 'mode': 0.0},
            'calm': {'energy': 0.3, 'valence': 0.6, 'tempo': 0.3},
            'serenity': {'energy': 0.2, 'valence': 0.6, 'acousticness': 0.7, 'tempo': 0.3},
            'excitement': {'energy': 0.8, 'valence': 0.8, 'tempo': 0.8},
            'excited': {'energy': 0.8, 'valence': 0.8, 'tempo': 0.8},
            'pride': {'valence': 0.8, 'energy': 0.6, 'tempo': 0.6},
            'nothing': {'energy': 0.5, 'valence': 0.5, 'tempo': 0.5},
            'neutral': {'energy': 0.5, 'valence': 0.5, 'tempo': 0.5},
            'nostalgia': {'valence': 0.6, 'energy': 0.4, 'acousticness': 0.6},
            'amusement': {'valence': 0.8, 'energy': 0.6},
            'humourous': {'valence': 0.8, 'energy': 0.6},
            'confusion': {'valence': 0.4, 'energy': 0.5, 'tempo': 0.5},
            'hope': {'valence': 0.7, 'energy': 0.6, 'mode': 1.0},
            'anxiety': {'energy': 0.7, 'valence': 0.3, 'tempo': 0.7},
            'apprehension': {'energy': 0.6, 'valence': 0.3, 'mode': 0.0},
            'contentment': {'valence': 0.8, 'energy': 0.4, 'mode': 1.0},
            'inspiration': {'valence': 0.8, 'energy': 0.6, 'mode': 1.0},
            'ambition': {'energy': 0.8, 'valence': 0.7, 'tempo': 0.7},
            'motivation': {'energy': 0.8, 'valence': 0.7, 'tempo': 0.7},
            'passion': {'energy': 0.8, 'valence': 0.7, 'tempo': 0.7},
            'love': {'valence': 0.8, 'energy': 0.5, 'mode': 1.0},
            'awe': {'valence': 0.7, 'energy': 0.5, 'acousticness': 0.6},
            'wonder': {'valence': 0.7, 'energy': 0.5, 'acousticness': 0.6},
            'chaotic': {'energy': 0.9, 'loudness': 0.8, 'tempo': 0.8},
            'electric': {'energy': 0.9, 'loudness': 0.8, 'tempo': 0.9},
            'wild': {'energy': 0.9, 'loudness': 0.8, 'tempo': 0.8},
            'grief': {'valence': 0.1, 'energy': 0.2, 'tempo': 0.2, 'mode': 0.0},
            'jealous': {'valence': 0.3, 'energy': 0.6, 'mode': 0.0},
            'guilt': {'valence': 0.2, 'energy': 0.3, 'mode': 0.0},
            'shame': {'valence': 0.1, 'energy': 0.3, 'mode': 0.0},
            'gratitude': {'valence': 0.8, 'energy': 0.5, 'mode': 1.0},
            'empathy': {'valence': 0.6, 'energy': 0.5},
            'curiosity': {'valence': 0.7, 'energy': 0.5, 'tempo': 0.6},
            'trust': {'valence': 0.8, 'energy': 0.5, 'mode': 1.0},
            'remorse': {'valence': 0.2, 'energy': 0.3, 'mode': 0.0},
            'regret': {'valence': 0.2, 'energy': 0.3, 'mode': 0.0},
            'loneliness': {'valence': 0.2, 'energy': 0.3, 'acousticness': 0.7},
            'boredom': {'valence': 0.4, 'energy': 0.2, 'tempo': 0.3},
            'relaxed': {'energy': 0.2, 'valence': 0.6, 'acousticness': 0.7}
        }
        matched_keywords = []
        for keyword, feature_map in keyword_mappings.items():
            if keyword in query_text:
                matched_keywords.append(keyword)
                if self.track_embeddings.shape[1] >= 11:
                    feature_map_positions = {
                        'danceability': 0,
                        'energy': 1,
                        'key': 2,
                        'loudness': 3,
                        'mode': 4,
                        'speechiness': 5,
                        'acousticness': 6,
                        'instrumentalness': 7,
                        'liveness': 8,
                        'valence': 9,
                        'tempo': 10,
                         'arousal': 11,
                        'intensity': 12,
                        'pleasantness': 13,
                        'complexity': 14,
                        'depth': 15,
                        'brightness': 16,      
                        'saturation': 17,      
                        'color_temperature': 18,  
                        'visual_complexity': 19,  
                        'emotional_weight': 20,  
                        'nostalgic_factor': 21,   
                        'social_context': 22,    
                        'attention_level': 23,    
                        'lyrical_relevance': 24, 
                        'cultural_context': 25,   
                        'movement_dynamic': 26,   
                        'contrast_level': 27,    
                        'emotional_transition': 28, 
                        'ambient_noise': 29,     
                        'personal_resonance': 30,
                        
                    }
                    for feature, value in feature_map.items():
                        if feature in feature_map_positions:
                            feature_vector[feature_map_positions[feature]] = value
                else:
                    i = 0
                    for feature, value in feature_map.items():
                        if i < self.track_embeddings.shape[1]:
                            feature_vector[i] = value
                            i += 1
        print(f"Matched keywords: {matched_keywords}")
        if not matched_keywords:
            print("No keywords matched. Using default balanced vector.")
            feature_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            feature_vector = feature_vector[:self.track_embeddings.shape[1]]
        return feature_vector

    def recommend_by_similar_tracks(self, track_ids: List[str], num_recommendations: int = 5) -> List[Dict]:
        try:
            print("\n=== SIMILAR TRACKS RECOMMENDATION SEARCH ===")
            print(f"Seed track IDs: {track_ids}")
            seed_tracks = self.tracks_df[self.tracks_df['track_id'].isin(track_ids)]
            if len(seed_tracks) == 0:
                print("No matching tracks found in the dataset.")
                return []
            print(f"Found {len(seed_tracks)} matching tracks in the dataset.")
            track_indices = seed_tracks['track_idx'].values
            centroid = np.mean(self.track_embeddings[track_indices], axis=0)
            centroid = centroid.reshape(1, -1).astype(np.float32)
            distances, indices = self.embedding_index.search(centroid, num_recommendations + len(track_indices))
            recommendations = []
            seed_track_indices = set(track_indices)
            for i, idx in enumerate(indices[0]):
                if idx < len(self.tracks_df) and idx not in seed_track_indices:
                    track = self.tracks_df.iloc[idx]
                    track_name = track['name'] if 'name' in track else track['artists']
                    print(f"Match: Track='{track_name}', artist='{track['artists']}', distance={distances[0][i]:.4f}")
                    recommendations.append({
                        'track_idx': int(track['track_idx']),
                        'track_id': track['track_id'],
                        'track_name': track_name,
                        'artist': track['artists'],
                        'album_name': track.get('album_name', 'Unknown'),
                        'genre': track.get('track_genre', 'Unknown'),
                        'duration_ms': int(track.get('duration_ms', 0)),
                        'popularity': int(track.get('popularity', 0)),
                        'distance': float(distances[0][i]),
                        'similarity': float(1.0 - distances[0][i] / (distances[0].max() + 1e-5))
                    })
                    if len(recommendations) >= num_recommendations:
                        break
            print("=== END SIMILAR TRACKS RECOMMENDATION SEARCH ===\n")
            return recommendations
        except Exception as e:
            logger.error(f"Error recommending by similar tracks: {e}")
            print(f"Exception during similar tracks recommendation: {e}")
            return []

    def get_track_info(self, track_id: str) -> Dict:
        try:
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].iloc[0]
            track_name = track['name'] if 'name' in track else track['artists']
            return {
                'track_idx': int(track['track_idx']),
                'track_id': track['track_id'],
                'track_name': track_name,
                'artist': track['artists'],
                'album_name': track.get('album_name', 'Unknown'),
                'genre': track.get('track_genre', 'Unknown'),
                'duration_ms': int(track.get('duration_ms', 0)),
                'popularity': int(track.get('popularity', 0)),
                'danceability': float(track.get('danceability', 0)),
                'energy': float(track.get('energy', 0)),
                'valence': float(track.get('valence', 0)),
                'tempo': float(track.get('tempo', 0)),
                'mode': int(track.get('mode', 0))
            }
        except Exception as e:
            logger.error(f"Error getting track info: {e}")
            return {}
            
    # Utility function to extract features from image
    def extract_image_features(self, image_data):
        """
        Extract features from an image that can influence music recommendations.
        In a production system, this would use computer vision libraries.
        
        Args:
            image_data: Binary image data or path to image file
            
        Returns:
            Dictionary of extracted image features
        """
        try:
            logger.info("Extracting features from image")
            
            # Placeholder - in a real implementation, would use CV libraries
            # like OpenCV, PIL, or deep learning models
            
            # Example placeholder return with default values
            features = {
                'brightness': 0.6,         # 0-1 scale
                'saturation': 0.5,         # 0-1 scale
                'color_temperature': 0.5,  # 0-1 scale (cool to warm)
                'visual_complexity': 0.5,  # 0-1 scale
                'movement_dynamic': 0.5,   # 0-1 scale (still to dynamic)
                'contrast_level': 0.5      # 0-1 scale
            }
            
            logger.info(f"Extracted image features: {features}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}", exc_info=True)
            return {}