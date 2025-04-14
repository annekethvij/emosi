"""
Constants used throughout the EMOSI system.
"""

EMOTION_CATEGORIES = [
    "happy", "sad", "calm", "excited", "angry", "relaxed", 
    "fearful", "surprised", "neutral"
]

MUSIC_EMOTION_FEATURES = {
    "happy": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "sad": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "low"},
    "calm": {"tempo": "slow", "mode": "major", "energy": "low", "valence": "medium"},
    "excited": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "angry": {"tempo": "fast", "mode": "minor", "energy": "high", "valence": "low"},
    "relaxed": {"tempo": "medium", "mode": "major", "energy": "low", "valence": "medium"},
    "fearful": {"tempo": "variable", "mode": "minor", "energy": "medium", "valence": "low"},
    "surprised": {"tempo": "variable", "mode": "variable", "energy": "high", "valence": "variable"},
    "neutral": {"tempo": "medium", "mode": "variable", "energy": "medium", "valence": "medium"}
}

QUESTIONNAIRE = [
    {
        "question": "How would you describe your current mood?",
        "options": [
            "Happy and cheerful", 
            "Sad or melancholic", 
            "Calm and peaceful", 
            "Excited or energetic",
            "Angry or frustrated",
            "Relaxed",
            "Fearful or anxious",
            "Surprised",
            "Neutral"
        ]
    },
    {
        "question": "What type of energy level are you looking for in music right now?",
        "options": [
            "High energy",
            "Medium energy",
            "Low energy/relaxing"
        ]
    },
    {
        "question": "What's the purpose of your Instagram story?",
        "options": [
            "Celebrating a happy moment",
            "Sharing a meaningful/serious event",
            "Just a casual update",
            "Travel or adventure",
            "Reflective/thoughtful moment"
        ]
    }
] 