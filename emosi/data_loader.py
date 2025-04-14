import os
import numpy as np
import pandas as pd
import logging
from typing import Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from .constants import EMOTION_CATEGORIES

logger = logging.getLogger(__name__)

class SpotifyDataLoader:
    def __init__(self, data_path: str = None, download_if_missing: bool = True):
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_path, exist_ok=True)
        logger.info(f"Data path set to: {self.data_path}")
        self.dataset_file = os.path.join(self.data_path, "data.csv")
        self.dataset_w_genres_file = os.path.join(self.data_path, "data_w_genres.csv")
        self.dataset_by_genres_file = os.path.join(self.data_path, "data_by_genres.csv")
        self.dataset_by_artists_file = os.path.join(self.data_path, "data_by_artist.csv")
        self.dataset_by_year_file = os.path.join(self.data_path, "data_by_year.csv")
        logger.info(f"Will check for primary dataset at: {self.dataset_file}")
        logger.info(f"Alternative dataset paths: {self.dataset_w_genres_file}, {self.dataset_by_genres_file}")
        self.processed_dir = os.path.join(self.data_path, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.tracks_file = os.path.join(self.processed_dir, "spotify_tracks_processed.csv")
        self.track_emotion_vectors_file = os.path.join(self.processed_dir, "track_emotion_vectors.npy")
        logger.info(f"Processed tracks file path: {self.tracks_file}")
        logger.info(f"Processed vectors file path: {self.track_emotion_vectors_file}")
        self.embedding_model = None
        self.track_embeddings_file = os.path.join(self.processed_dir, "track_embeddings.npy")
        self.tracks_df = None
        self.track_emotion_vectors = None
        self.track_embeddings = None
        self._load_or_process_data()
        self._load_or_create_embeddings()

    def _load_or_process_data(self):
        if os.path.exists(self.tracks_file) and os.path.exists(self.track_emotion_vectors_file):
            logger.info(f"Found processed files at {self.tracks_file}. Loading...")
            self._load_processed_data()
        elif os.path.exists("/root/test/data/data.csv"):
            logger.info("Found /root/test/data/data.csv directly. Using this file...")
            self.dataset_file = "/root/test/data/data.csv"
            self._process_raw_data()
            self._save_processed_data()
        elif os.path.exists(self.dataset_file):
            logger.info(f"Found dataset at {self.dataset_file}. Processing...")
            self._process_raw_data()
            self._save_processed_data()
        elif os.path.exists(self.dataset_w_genres_file):
            logger.info(f"Found data_w_genres.csv at {self.dataset_w_genres_file}. Processing...")
            self.dataset_file = self.dataset_w_genres_file
            self._process_raw_data()
            self._save_processed_data()
        elif os.path.exists(self.dataset_by_genres_file):
            logger.info(f"Found genre-level data at {self.dataset_by_genres_file}. Processing...")
            self.dataset_file = self.dataset_by_genres_file
            self._process_raw_data()
            self._save_processed_data()
        else:
            try:
                logger.info(f"Listing all files in {self.data_path}:")
                if os.path.exists(self.data_path):
                    for file in os.listdir(self.data_path):
                        logger.info(f"  - {file}")
                else:
                    logger.info(f"Data path {self.data_path} doesn't exist")
                root_test_data = "/root/test/data"
                if os.path.exists(root_test_data):
                    logger.info(f"Listing all files in {root_test_data}:")
                    for file in os.listdir(root_test_data):
                        logger.info(f"  - {file}")
                else:
                    logger.info(f"Path {root_test_data} doesn't exist")
            except Exception as e:
                logger.error(f"Error listing files: {e}")
            logger.error(f"No dataset file found in {self.data_path} or at /root/test/data/data.csv")
            raise FileNotFoundError(f"Dataset not found. Please place a valid Spotify dataset CSV in {self.data_path}")

    def _load_processed_data(self):
        try:
            self.tracks_df = pd.read_csv(self.tracks_file)
            logger.info(f"Loaded DataFrame columns: {list(self.tracks_df.columns)}")
            logger.info(f"Check if 'name' column exists: {'name' in self.tracks_df.columns}")
            self.track_emotion_vectors = np.load(self.track_emotion_vectors_file)
            logger.info(f"Loaded data: {len(self.tracks_df)} tracks")
            print("\n=== DATASET SAMPLE ===")
            print(f"Track dataframe sample (first 5 rows):")
            print(self.tracks_df.head(5).to_string())
            print("\nTrack emotion vectors shape:", self.track_emotion_vectors.shape)
            print("Sample emotion vector for first track:", self.track_emotion_vectors[0])
            print("=== END DATASET SAMPLE ===\n")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise

    def _process_raw_data(self):
        try:
            logger.info(f"Reading Spotify track data from {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
            logger.info(f"Raw dataset columns: {list(df.columns)}")
            df['original_name'] = df['name'].copy() if 'name' in df.columns else None
            column_mapping = {
                'artists': 'artists',
                'name': 'name',
                'id': 'track_id',
                'popularity': 'popularity',
                'danceability': 'danceability',
                'energy': 'energy',
                'key': 'key',
                'loudness': 'loudness',
                'mode': 'mode',
                'speechiness': 'speechiness',
                'acousticness': 'acousticness',
                'instrumentalness': 'instrumentalness',
                'liveness': 'liveness',
                'valence': 'valence',
                'tempo': 'tempo',
                'duration_ms': 'duration_ms',
                'genres': 'track_genre',
                'year': 'year'
            }
            renamed_cols = {}
            for orig_col, expected_col in column_mapping.items():
                if orig_col in df.columns:
                    renamed_cols[orig_col] = expected_col
            logger.info(f"Columns being renamed: {renamed_cols}")
            df = df.rename(columns=renamed_cols)
            df['track_idx'] = range(len(df))
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean())
            logger.info(f"Processed dataset columns: {list(df.columns)}")
            logger.info(f"Check if 'name' column exists in processed data: {'name' in df.columns}")
            self.tracks_df = df
            logger.info("Generating emotion vectors for tracks...")
            self._generate_track_emotion_vectors()
            logger.info(f"Processed data: {len(self.tracks_df)} tracks")
        except Exception as e:
            logger.error(f"Error processing raw data: {e}")
            raise

    def _generate_track_emotion_vectors(self):
        num_tracks = len(self.tracks_df)
        vector_dim = len(EMOTION_CATEGORIES)
        self.track_emotion_vectors = np.zeros((num_tracks, vector_dim))
        feature_mappings = {
            'danceability': {
                'high': {'happy': 0.7, 'excited': 0.8, 'relaxed': 0.3},
                'medium': {'neutral': 0.5, 'calm': 0.3, 'happy': 0.3},
                'low': {'sad': 0.6, 'calm': 0.5, 'fearful': 0.2}
            },
            'energy': {
                'high': {'excited': 0.8, 'angry': 0.6, 'surprised': 0.4},
                'medium': {'happy': 0.5, 'neutral': 0.5},
                'low': {'sad': 0.6, 'calm': 0.7, 'relaxed': 0.6}
            },
            'valence': {
                'high': {'happy': 0.9, 'excited': 0.6},
                'medium': {'relaxed': 0.6, 'neutral': 0.5, 'calm': 0.4},
                'low': {'sad': 0.8, 'angry': 0.6, 'fearful': 0.5}
            },
            'acousticness': {
                'high': {'calm': 0.7, 'relaxed': 0.6, 'sad': 0.4},
                'medium': {'neutral': 0.5, 'happy': 0.3},
                'low': {'excited': 0.5, 'surprised': 0.4}
            },
            'tempo': {
                'high': {'excited': 0.7, 'angry': 0.6, 'surprised': 0.5},
                'medium': {'happy': 0.5, 'neutral': 0.5},
                'low': {'sad': 0.6, 'calm': 0.7, 'relaxed': 0.6}
            }
        }
        thresholds = {
            'danceability': {'low': 0.3, 'high': 0.7},
            'energy': {'low': 0.3, 'high': 0.7},
            'valence': {'low': 0.3, 'high': 0.7},
            'acousticness': {'low': 0.3, 'high': 0.7},
            'tempo': {'low': 90, 'high': 140}
        }
        tempo_min = self.tracks_df['tempo'].min()
        tempo_max = self.tracks_df['tempo'].max()
        self.tracks_df['tempo_normalized'] = (self.tracks_df['tempo'] - tempo_min) / (tempo_max - tempo_min)
        for idx, track in tqdm(self.tracks_df.iterrows(), total=len(self.tracks_df), desc="Generating emotion vectors"):
            for feature, mapping in feature_mappings.items():
                feature_value = track[feature] if feature != 'tempo' else track['tempo_normalized']
                if feature_value <= thresholds[feature]['low']:
                    level = 'low'
                elif feature_value >= thresholds[feature]['high']:
                    level = 'high'
                else:
                    level = 'medium'
                for emotion, weight in mapping[level].items():
                    emotion_idx = EMOTION_CATEGORIES.index(emotion)
                    self.track_emotion_vectors[track['track_idx'], emotion_idx] += weight
        for idx, track in self.tracks_df.iterrows():
            track_idx = track['track_idx']
            if track['mode'] == 1:
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('happy')] *= 1.2
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('excited')] *= 1.1
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('relaxed')] *= 1.1
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('sad')] *= 0.8
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('fearful')] *= 0.9
            else:
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('sad')] *= 1.2
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('fearful')] *= 1.1
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('angry')] *= 1.1
                self.track_emotion_vectors[track_idx, EMOTION_CATEGORIES.index('happy')] *= 0.8
        row_sums = self.track_emotion_vectors.sum(axis=1)
        self.track_emotion_vectors = self.track_emotion_vectors / row_sums[:, np.newaxis]

    def _save_processed_data(self):
        try:
            self.tracks_df.to_csv(self.tracks_file, index=False)
            np.save(self.track_emotion_vectors_file, self.track_emotion_vectors)
            logger.info(f"Saved processed data to {self.processed_dir}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    def _load_or_create_embeddings(self):
        if os.path.exists(self.track_embeddings_file):
            logger.info("Loading pre-computed track embeddings...")
            self.track_embeddings = np.load(self.track_embeddings_file)
        else:
            logger.info("Computing track embeddings for RAG...")
            self._compute_track_embeddings()
            np.save(self.track_embeddings_file, self.track_embeddings)

    def _compute_track_embeddings(self):
        try:
            track_texts = []
            for _, track in tqdm(self.tracks_df.iterrows(), total=len(self.tracks_df), desc="Creating track descriptions"):
                text = f"Track: {track['name'] if 'name' in track else track['artists']} by {track['artists']}. "
                if 'album_name' in track:
                    text += f"Album: {track['album_name']}. "
                if 'track_genre' in track:
                    text += f"Genre: {track['track_genre']}. "
                text += f"Features: danceability {track['danceability']:.2f}, "
                text += f"energy {track['energy']:.2f}, "
                text += f"valence {track['valence']:.2f}, "
                text += f"acousticness {track['acousticness']:.2f}, "
                text += f"tempo {track['tempo']:.1f}."
                track_texts.append(text)
            logger.info("Loading SentenceTransformer model...")
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                logger.info("Using a simplified embedding approach...")
                self.embedding_model = None
            if self.embedding_model:
                logger.info("Computing track embeddings with SentenceTransformer...")
                self.track_embeddings = self.embedding_model.encode(
                    track_texts,
                    show_progress_bar=True,
                    batch_size=32,
                    convert_to_numpy=True
                )
            else:
                logger.info("Using audio features as embeddings...")
                feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                               'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo']
                scaler = MinMaxScaler()
                normalized_features = scaler.fit_transform(self.tracks_df[feature_cols])
                self.track_embeddings = normalized_features
            logger.info(f"Created embeddings with shape: {self.track_embeddings.shape}")
        except Exception as e:
            logger.error(f"Error computing track embeddings: {e}")
            logger.info("Falling back to basic audio feature embeddings...")
            feature_cols = ['danceability', 'energy', 'valence', 'acousticness']
            self.track_embeddings = self.tracks_df[feature_cols].values

    def get_tracks(self) -> pd.DataFrame:
        return self.tracks_df

    def get_track_emotion_vectors(self) -> np.ndarray:
        return self.track_emotion_vectors

    def get_track_embeddings(self) -> np.ndarray:
        return self.track_embeddings

