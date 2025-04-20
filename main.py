#!/usr/bin/env python
"""
EMOSI: Emotion-based Music Selection Interface (Spotify Version)

This script serves as an entry point for the EMOSI system.

Usage:
    python main.py --mode=text --text="Happy energetic dance music" --year-cutoff=2015
    python main.py --mode=image --image=path/to/image.jpg
    
For more options:
    python main.py --help
"""

import os
import sys
import argparse
import logging
import json
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the nested emosi package
from emosi.facade import EmosiFacade
from emosi.utils import format_recommendation_output

def run_cli_demo():
    """Run the command line interface demo."""
    # Add your CLI code here - copy from the original file
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EMOSI: Emotion-based Music Selection Interface')
    parser.add_argument('--text', type=str, help='Text query for music recommendation')
    parser.add_argument('--mode', type=str, default='text', choices=['image', 'text', 'combined'], 
                        help='Mode to run: image-based, text-based, or combined')
    parser.add_argument('--year-cutoff', type=int, default=2010, 
                        help='Prefer songs released after this year (default: 2010)')
    parser.add_argument('--num-recommendations', type=int, default=5,
                        help='Number of recommendations to return')
    parser.add_argument('--image', type=str, help='Path to image for emotion-based recommendation')
    
    args = parser.parse_args()
    
    # Initialize the EMOSI system with dummy mode for testing
    print("\n" + "="*80)
    print("EMOSI: Emotion-based Music Selection Interface")
    print("="*80)
    
    try:
        emosi = EmosiFacade(use_dummy=True)
        
        if args.mode == 'text' and args.text:
            print(f"Running text query: '{args.text}' with year cutoff: {args.year_cutoff}")
            
            recommendations = emosi.recommend_by_text(
                query_text=args.text, 
                num_recommendations=args.num_recommendations,
                year_cutoff=args.year_cutoff
            )
            
            print("\nRecommended songs:")
            print(format_recommendation_output(recommendations))
        elif args.mode == 'image' and args.image:
            print(f"Running image-based recommendation using: '{args.image}'")
            
            dominant_emotion, emotion_scores, recommendations = emosi.run_image_based_recommendation(
                image_path=args.image,
                num_recommendations=args.num_recommendations
            )
            
            print(f"\nDetected emotion: {dominant_emotion}")
            print("Emotion scores:")
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {score:.3f}")
            
            print("\nRecommended songs:")
            print(format_recommendation_output(recommendations))
        else:
            print("Please provide required arguments:")
            print("  For text mode: --mode=text --text='your query'")
            print("  For image mode: --mode=image --image=path/to/image.jpg")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("Thank you for using EMOSI!")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_cli_demo()
