import os
import logging
from typing import Dict, List

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from .recommender import SpotifyRecommender
from .constants import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)

class RAGEnhancer:
    """
    RAG (Retrieval Augmented Generation) for music recommendations.
    """
    def __init__(self, spotify_recommender: SpotifyRecommender):
        self.recommender = spotify_recommender
        self.tracks_df = spotify_recommender.tracks_df
        self.spotify_api = None
        try:
            self._connect_to_spotify_api()
        except Exception as e:
            logger.warning(f"Could not connect to Spotify API: {e}")
    
    def _connect_to_spotify_api(self):
        """ OPTIONAL, We All can decide on this one latter, if or not to Connect to the Spotify API using environment variables."""
        try:
            client_id = os.environ.get('SPOTIFY_CLIENT_ID')
            client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
            if client_id and client_secret:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                self.spotify_api = spotipy.Spotify(
                    client_credentials_manager=client_credentials_manager
                )
                logger.info("Connected to Spotify API successfully")
            else:
                logger.warning("Spotify API credentials not found in environment variables")
        except Exception as e:
            logger.error(f"Error connecting to Spotify API: {e}")
            self.spotify_api = None
            
def _generate_slider_based_recommendation_reason(self, recommendation: Dict, emotion_sliders: Dict[str, float]) -> str:
    """
    Generate recommendation reasons based on emotion slider inputs.
    
    Args:
        recommendation: The track recommendation
        emotion_sliders: Dictionary mapping emotions to intensity values (0-10)
        
    Returns:
        String explaining why the track was recommended
    """
    track_idx = recommendation['track_idx']
    track = self.tracks_df.iloc[track_idx]
    
    # Sort emotions by intensity to find primary emotions
    top_emotions = sorted(
        [(emotion, intensity) for emotion, intensity in emotion_sliders.items() if intensity > 0], 
        key=lambda x: x[1], 
        reverse=True
    )[:3]  # Get top 3 emotions
    
    if not top_emotions:
        return "This track was recommended based on your preferences."
    
    # If there's only one dominant emotion with high intensity
    if len(top_emotions) == 1 or (top_emotions[0][1] > 7 and top_emotions[0][1] > top_emotions[1][1] * 2):
        dominant_emotion = top_emotions[0][0]
        reason = f"This track was recommended because it strongly matches your {dominant_emotion} feeling. "
        
        # Add specific audio feature explanation based on the emotion
        if dominant_emotion.lower() in ['happiness', 'happy', 'joy']:
            if track['valence'] > 0.7:
                reason += f"It has a very positive tone (valence: {track['valence']:.2f})."
            elif track['energy'] > 0.7:
                reason += f"It has high energy ({track['energy']:.2f}) that amplifies positive feelings."
        elif dominant_emotion.lower() in ['sadness', 'sad']:
            if track['valence'] < 0.4:
                reason += f"It has a more melancholic tone (valence: {track['valence']:.2f})."
            elif track['mode'] == 0:
                reason += f"It's in a minor key, which often evokes this emotion."
        elif dominant_emotion.lower() in ['fear', 'fearful', 'anxiety', 'apprehension']:
            if track['valence'] < 0.4 and track['energy'] > 0.5:
                reason += f"It combines tense energy with lower valence, creating the right atmosphere."
        elif dominant_emotion.lower() in ['calm', 'serenity', 'relaxed', 'contentment']:
            if track['energy'] < 0.4:
                reason += f"It has low energy ({track['energy']:.2f}) that creates a calm atmosphere."
            elif track['acousticness'] > 0.6:
                reason += f"It's acoustically rich ({track['acousticness']:.2f}), perfect for relaxation."
        elif dominant_emotion.lower() in ['surprised', 'surprise']:
            if track['liveness'] > 0.6:
                reason += f"It has unexpected live elements that create a sense of surprise."
        elif dominant_emotion.lower() in ['disgust']:
            if track['valence'] < 0.3 and track['energy'] > 0.5:
                reason += f"It has dissonant qualities that match this feeling."
        elif dominant_emotion.lower() in ['anger', 'angry']:
            if track['energy'] > 0.8:
                reason += f"It has intense energy ({track['energy']:.2f}) that channels this emotion."
        elif dominant_emotion.lower() in ['excitement', 'excited']:
            if track['tempo'] > 120 and track['energy'] > 0.7:
                reason += f"It has a driving tempo ({track['tempo']:.1f} BPM) with high energy."
        elif dominant_emotion.lower() in ['pride']:
            if track['valence'] > 0.6 and track['energy'] > 0.5:
                reason += f"It has a confident, positive tone that resonates with this feeling."
        elif dominant_emotion.lower() in ['nostalgia']:
            if track['acousticness'] > 0.6:
                reason += f"It has warm acoustic qualities that evoke memories."
        elif dominant_emotion.lower() in ['amusement', 'humourous']:
            if track['valence'] > 0.7 and track['energy'] > 0.5:
                reason += f"It has a playful combination of positivity and energy."
        elif dominant_emotion.lower() in ['hope']:
            if track['mode'] == 1 and track['valence'] > 0.5:
                reason += f"It uses major key with positive valence to inspire optimism."
        elif dominant_emotion.lower() in ['love']:
            if track['valence'] > 0.6 and track['acousticness'] > 0.5:
                reason += f"It has a warm, intimate quality perfect for this emotion."
        elif dominant_emotion.lower() in ['awe', 'wonder']:
            if track['acousticness'] > 0.6 and track['instrumentalness'] > 0.5:
                reason += f"It has expansive sonic qualities that create a sense of wonder."
        elif dominant_emotion.lower() in ['chaotic', 'electric', 'wild']:
            if track['energy'] > 0.8:
                reason += f"It has intense energy ({track['energy']:.2f}) that matches this feeling."
        elif dominant_emotion.lower() in ['grief']:
            if track['valence'] < 0.3 and track['tempo'] < 80:
                reason += f"It combines deep emotional tone with slow pace."
        elif dominant_emotion.lower() in ['confusion']:
            if track['mode'] == 0 and track['valence'] < 0.5:
                reason += f"It has tonal complexities that mirror this feeling."
        else:
            reason += f"Its audio profile closely matches this emotional signature."
    
    # If there are multiple emotions with similar intensities
    else:
        primary_emotion = top_emotions[0][0]
        secondary_emotion = top_emotions[1][0]
        reason = f"This track was recommended because it balances your {primary_emotion} and {secondary_emotion} feelings. "
        
        # Add explanation based on the track's audio features
        if track['valence'] > 0.7:
            reason += f"It has a positive tone (valence: {track['valence']:.2f}) "
        elif track['valence'] < 0.3:
            reason += f"It has a melancholic quality (valence: {track['valence']:.2f}) "
        
        if track['energy'] > 0.7:
            reason += f"with high energy ({track['energy']:.2f}), "
        elif track['energy'] < 0.3:
            reason += f"with gentle energy ({track['energy']:.2f}), "
        
        if track['tempo'] > 120:
            reason += f"and a driving tempo of {track['tempo']:.1f} BPM."
        elif track['tempo'] < 90:
            reason += f"and a relaxed tempo of {track['tempo']:.1f} BPM."
        else:
            reason += f"and balanced musical elements."
    
    # Add genre information if available
    if 'track_genre' in track and str(track['track_genre']) != 'nan' and str(track['track_genre']) != 'Unknown':
        reason += f" It belongs to the {track['track_genre']} genre."
        
    return reason

  
    
    def enhance_recommendations(self, recommendations: List[Dict], query_context: Dict) -> List[Dict]:
        """
        Enhance recommendations with additional context and information.
        """
        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = rec.copy()
            enhanced_rec['recommendation_reason'] = self._generate_recommendation_reason(rec, query_context)
            enhanced_rec['spotify_url'] = f"https://open.spotify.com/track/{rec['track_id']}"
            enhanced_rec['feature_summary'] = self._get_feature_summary(rec['track_idx'])
            if self.spotify_api:
                try:
                    track_info = self.spotify_api.track(rec['track_id'])
                    enhanced_rec['preview_url'] = track_info.get('preview_url')
                    enhanced_rec['album_image'] = track_info.get('album', {}).get('images', [{}])[0].get('url')
                except Exception as e:
                    logger.warning(f"Error fetching Spotify data for track {rec['track_id']}: {e}")
            enhanced_recommendations.append(enhanced_rec)
        return enhanced_recommendations
    
    def _generate_recommendation_reason(self, recommendation: Dict, query_context: Dict) -> str:
        track_idx = recommendation['track_idx']
        track = self.tracks_df.iloc[track_idx]
        reason = "This track was recommended because "
        if 'emotion' in query_context:
            emotion = query_context['emotion']
            if emotion in ['happy', 'excited']:
                if track['valence'] > 0.7:
                    reason += f"it has a very positive vibe (valence: {track['valence']:.2f})"
                elif track['energy'] > 0.7:
                    reason += f"it has high energy ({track['energy']:.2f}) that matches your {emotion} mood"
                elif track['danceability'] > 0.7:
                    reason += f"it's highly danceable ({track['danceability']:.2f}) and fits your {emotion} feeling"
                else:
                    reason += f"its overall musical profile matches a {emotion} emotion"
            elif emotion in ['sad', 'fearful']:
                if track['valence'] < 0.4:
                    reason += f"it has a more melancholic tone (valence: {track['valence']:.2f})"
                elif track['mode'] == 0:
                    reason += f"it's in a minor key, which often evokes {emotion} emotions"
                else:
                    reason += f"its overall musical profile matches a {emotion} emotion"
            elif emotion in ['calm', 'relaxed']:
                if track['energy'] < 0.4:
                    reason += f"it has low energy ({track['energy']:.2f}) that creates a {emotion} atmosphere"
                elif track['acousticness'] > 0.6:
                    reason += f"it's acoustically rich ({track['acousticness']:.2f}), creating a {emotion} sound"
                else:
                    reason += f"its tempo and energy levels match a {emotion} state"
            else:
                reason += f"its audio profile closely matches the {emotion} emotional signature"
        elif 'query_text' in query_context:
            query = query_context['query_text'].lower()
            if 'dance' in query or 'danceable' in query:
                reason += f"it has a danceability score of {track['danceability']:.2f}"
            elif 'energy' in query or 'energetic' in query:
                reason += f"it has an energy level of {track['energy']:.2f}"
            elif 'happy' in query or 'positive' in query:
                reason += f"it has a positive valence of {track['valence']:.2f}"
            elif 'sad' in query or 'melancholic' in query:
                reason += f"it has a more melancholic valence of {track['valence']:.2f}"
            elif 'acoustic' in query:
                reason += f"it has an acousticness level of {track['acousticness']:.2f}"
            elif 'tempo' in query or 'fast' in query or 'slow' in query:
                reason += f"it has a tempo of {track['tempo']:.1f} BPM"
            else:
                reason += f"its audio features closely match what you described in your query"
        elif 'track_ids' in query_context:
            reason += f"it has similar musical characteristics to the tracks you selected"
        if 'track_genre' in track and str(track['track_genre']) != 'nan' and str(track['track_genre']) != 'Unknown':
            reason += f". It belongs to the {track['track_genre']} genre."
        return reason
    
    def _get_feature_summary(self, track_idx: int) -> str:
        track = self.tracks_df.iloc[track_idx]
        def get_level(value, feature):
            if feature == 'tempo':
                if value < 80:
                    return "slow"
                elif value > 140:
                    return "very fast"
                elif value > 120:
                    return "fast"
                else:
                    return "moderate"
            else:
                if value < 0.3:
                    return "low"
                elif value > 0.7:
                    return "high"
                else:
                    return "moderate"
        key_names = ['C', 'C♯/D♭', 'D', 'D♯/E♭', 'E', 'F', 'F♯/G♭', 'G', 'G♯/A♭', 'A', 'A♯/B♭', 'B']
        key_name = key_names[track['key']] if 0 <= track['key'] < len(key_names) else 'Unknown'
        mode_name = "Major" if track['mode'] == 1 else "Minor"
        summary = f"This track has {get_level(track['danceability'], 'danceability')} danceability, "
        summary += f"{get_level(track['energy'], 'energy')} energy, and "
        summary += f"{get_level(track['valence'], 'valence')} positivity (valence). "
        summary += f"It's in the key of {key_name} {mode_name} with a {get_level(track['tempo'], 'tempo')} "
        summary += f"tempo of {track['tempo']:.1f} BPM."
        return summary

