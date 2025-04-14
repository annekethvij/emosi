import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def format_recommendation_output(recommendations: List[Dict]) -> str:
    if not recommendations:
        return "No recommendations found."

    output = []
    for i, rec in enumerate(recommendations, 1):
        try:
            logging.info(f"Formatting recommendation {i}: {rec}")
            
            track_name = rec.get('track_name', 'Unknown Track')
            artist = rec.get('artist', '')
            year = rec.get('release_year', rec.get('year', ''))
            year_str = f" ({year})" if year else ""
            if artist and artist != track_name:
                output.append(f"{i}. {track_name} - {artist}{year_str}")
            else:
                output.append(f"{i}. {track_name}{year_str}")
            
            if 'reason' in rec or 'recommendation_reason' in rec:
                reason = rec.get('reason', rec.get('recommendation_reason', ''))
                if reason and not reason.endswith(' '):
                    output.append(f"   Reason: {reason}")
            
            if 'genre' in rec and rec['genre']:
                genre = rec['genre']
                output.append(f"   Genre: {genre}")
            
            for feature in ['valence', 'energy', 'danceability']:
                if feature in rec:
                    output.append(f"   {feature.capitalize()}: {rec[feature]:.2f}")
            
            output.append("")
        except Exception as e:
            logging.error(f"Error formatting recommendation {i}: {e}")
            logging.error("Exception details:", exc_info=True)
            output.append(f"{i}. [Error displaying recommendation]")
            output.append("")
    
    return "\n".join(output)
