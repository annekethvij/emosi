"""
Constants used throughout the EMOSI system.
"""

EMOTION_CATEGORIES = [
    "happy", "sad", "calm", "excited", "angry", "relaxed", 
    "fearful", "surprised", "neutral",
    "happiness", "joy", "sadness", "fear", "surprise", "disgust",
    "anger", "calm", "serenity", "excitement", "pride",
    "nothing", "neutral", "nostalgia", "amusement", "humourous",
    "confusion", "hope", "anxiety", "apprehension", "contentment",
    "inspiration", "ambition", "motivation", "passion",
    "love", "awe", "wonder", "chaotic", "electric", "wild", "grief", "jealous", "guilt", "shame", "gratitide", "empathy",
    "curiosity", "trust", "remorse", "regret", "lonliness", "boredom"
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
    "neutral": {"tempo": "medium", "mode": "variable", "energy": "medium", "valence": "medium"},
    "joy": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "disgust": {"tempo": "slow", "mode": "minor", "energy": "medium", "valence": "low"},
    "serenity": {"tempo": "slow", "mode": "major", "energy": "low", "valence": "medium"},
    "pride": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "nothing": {"tempo": "medium", "mode": "variable", "energy": "low", "valence": "medium"},
    "nostalgia": {"tempo": "medium", "mode": "variable", "energy": "low", "valence": "medium"},
    "amusement": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "humourous": {"tempo": "variable", "mode": "major", "energy": "medium", "valence": "high"},
    "confusion": {"tempo": "variable", "mode": "variable", "energy": "medium", "valence": "low"},
    "hope": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "anxiety": {"tempo": "fast", "mode": "minor", "energy": "high", "valence": "low"},
    "apprehension": {"tempo": "medium", "mode": "minor", "energy": "medium", "valence": "low"},
    "contentment": {"tempo": "medium", "mode": "major", "energy": "low", "valence": "high"},
    "inspiration": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "ambition": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "motivation": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "passion": {"tempo": "fast", "mode": "major", "energy": "high", "valence": "high"},
    "love": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "awe": {"tempo": "slow", "mode": "major", "energy": "medium", "valence": "high"},
    "wonder": {"tempo": "slow", "mode": "major", "energy": "medium", "valence": "high"},
    "chaotic": {"tempo": "fast", "mode": "variable", "energy": "high", "valence": "variable"},
    "electric": {"tempo": "fast", "mode": "variable", "energy": "very high", "valence": "high"},
    "wild": {"tempo": "fast", "mode": "variable", "energy": "very high", "valence": "variable"},
     "grief": {"tempo": "very slow", "mode": "minor", "energy": "low", "valence": "very low"},
    "jealous": {"tempo": "medium", "mode": "minor", "energy": "medium", "valence": "low"},
    "guilt": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "low"},
    "shame": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "very low"},
    "gratitude": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "empathy": {"tempo": "variable", "mode": "variable", "energy": "medium", "valence": "variable"},
    "curiosity": {"tempo": "medium", "mode": "variable", "energy": "medium", "valence": "medium-high"},
    "trust": {"tempo": "medium", "mode": "major", "energy": "medium", "valence": "high"},
    "remorse": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "low"},
    "regret": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "low"},
    "loneliness": {"tempo": "slow", "mode": "minor", "energy": "low", "valence": "low"},
    "boredom": {"tempo": "slow", "mode": "variable", "energy": "very low", "valence": "low"}
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
            "Neutral",
            "Serenity", "Pride",
            "Nothing", "Neutral", "Nostalgia", "Amusement", "Humourous",
            "Confusion", "Hope", "Apprehension", "Contentment",
            "Inspiration", "Ambition", "Motivation", "Passion",
            "Love", "Awe", "Wonder", "Chaotic", "Electric", "Wild","grief", "jealous", "guilt", "shame", "gratitide", "empathy",
            "curiosity", "trust", "remorse", "regret", "lonliness", "boredom"
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