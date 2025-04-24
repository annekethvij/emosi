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
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path
from PIL import Image

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the nested emosi package
from emosi.facade import EmosiFacade
from emosi.utils import format_recommendation_output

def setup_logging(output_dir):
    """Set up logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"emosi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def process_csv_emotion_data(csv_file_path):
    """
    Process slider-based emotion data from survey responses CSV.
    Returns dictionaries mapping names and image filenames to emotion scores and additional data.
    """
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Create mappings for efficient lookups
    image_emotions = {}     # Maps Image_Name to data
    name_emotions = {}      # Maps Name to data
    clean_name_emotions = {}  # Maps cleaned names to data
    
    # Map between CSV columns and emotion categories
    emotion_columns = {
        'Emotion Felt_1': 'happiness',
        'Emotion Felt_2': 'sadness',
        'Emotion Felt_3': 'fear',
        'Emotion Felt_4': 'surprise',
        'Emotion Felt_5': 'disgust',
        'Emotion Felt_6': 'anger',
        'Emotion Felt_7': 'serenity',
        'Emotion Felt_8': 'excitement',
        'Emotion Felt_9': 'pride',
        'Emotion Felt_10': 'neutral',
        'Emotion Felt_11': 'nostalgia',
        'Emotion Felt_12': 'amusement',
        'Emotion Felt_13': 'confusion',
        'Emotion Felt_14': 'hope',
        'Emotion Felt_15': 'anxiety',
        'Emotion Felt_16': 'contentment',
        'Emotion Felt_17': 'inspiration',
        'Emotion Felt_18': 'love',
        'Emotion Felt_19': 'awe',
        'Emotion Felt_20': 'chaotic'
    }
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        if pd.notna(row.get('Name')) and pd.notna(row.get('Image_Name')):
            emotions = {}
            
            # Extract emotion scores for each emotion category
            for col, emotion in emotion_columns.items():
                if pd.notna(row.get(col)) and row.get(col) != '':
                    try:
                        score = float(row.get(col)) / 10.0  # Normalize to 0-1 range
                        emotions[emotion] = score
                    except (ValueError, TypeError):
                        pass  # Skip if conversion fails
            
            # Get additional description text if available
            additional_text = None
            if pd.notna(row.get('Q12')) and row.get('Q12') not in ['N/A', 'N/a', 'NA', 'n/a']:
                additional_text = row.get('Q12')
            
            # Create data record
            data = {
                'emotions': emotions,
                'description': additional_text,
                'name': row['Name'].strip(),
                'image_name': row['Image_Name']
            }
            
            # Store data under multiple keys for different lookup strategies
            if emotions:
                # Store by original image name
                image_emotions[row['Image_Name']] = data
                
                # Store by person's name
                name_emotions[row['Name'].strip()] = data
                
                # Store by cleaned name (lowercase, no spaces)
                clean_name = row['Name'].strip().lower().replace(' ', '').replace('_', '')
                clean_name_emotions[clean_name] = data
                
                # Store by file name (without extension) for additional matching
                file_base = row['Image_Name'].split('.')[0]
                image_emotions[file_base] = data
                
                # Store with lowercase image name
                image_emotions[row['Image_Name'].lower()] = data
    
    return image_emotions, name_emotions, clean_name_emotions

def find_matching_emotion_data(file_name, image_emotions, name_emotions, clean_name_emotions):
    """
    Find matching emotion data for a given filename using multiple strategies.
    
    Args:
        file_name: The filename to match
        image_emotions: Dictionary mapping image names to data
        name_emotions: Dictionary mapping person names to data
        clean_name_emotions: Dictionary mapping cleaned names to data
        
    Returns:
        The matching data or None
    """
    # 1. Direct match with image name
    if file_name in image_emotions:
        return image_emotions[file_name]
    
    # 2. Try lowercase
    if file_name.lower() in image_emotions:
        return image_emotions[file_name.lower()]
    
    # 3. Try without extension
    base_name = os.path.splitext(file_name)[0]
    if base_name in image_emotions:
        return image_emotions[base_name]
    
    # 4. Try with lowercase base name
    if base_name.lower() in image_emotions:
        return image_emotions[base_name.lower()]
    
    # 5. Try cleaning the filename and match against person names
    # This handles cases like "Aayushi_ Singh.jpeg" matching "Aayushi Singh"
    clean_file = base_name.lower().replace('_', '').replace(' ', '')
    if clean_file in clean_name_emotions:
        return clean_name_emotions[clean_file]
    
    # 6. Try to extract name from filename and match
    for name in name_emotions:
        clean_name = name.lower().replace(' ', '').replace('_', '')
        if clean_name in clean_file or clean_file in clean_name:
            return name_emotions[name]
    
    # 7. Last resort - try substring matching
    for img_name in image_emotions:
        if isinstance(img_name, str) and len(img_name) > 3:
            if img_name.lower() in file_name.lower() or file_name.lower() in img_name.lower():
                return image_emotions[img_name]
    
    # No match found
    return None

def analyze_and_recommend(emosi, image_path, emotion_data=None, num_recommendations=10, year_cutoff=2015, logger=None):
    """
    Generate recommendations using different modalities.
    
    Args:
        emosi: EmosiFacade instance
        image_path: Path to the image file
        emotion_data: Dictionary containing emotion slider data and description
        num_recommendations: Number of recommendations to generate
        year_cutoff: Cutoff year for preferring newer songs
        logger: Logger instance to use
        
    Returns:
        Dictionary containing recommendations for different modalities
    """
    results = {}
    
    # Get image name for logging
    image_name = os.path.basename(image_path)
    
    # Use provided logger or create a basic one
    if logger is None:
        logger = logging.getLogger()
    
    # 1. Image-based recommendations
    logger.info(f"Generating image-based recommendations for {image_name}")
    try:
        img_dominant_emotion, img_emotion_scores, img_recommendations = emosi.run_image_based_recommendation(
            image_path=image_path,
            num_recommendations=num_recommendations
        )
        
        # Get the full image analysis text if available
        image_analysis = emosi.image_detector.get_last_analysis() if hasattr(emosi.image_detector, 'get_last_analysis') else "Image analysis not available"
        
        results['image'] = {
            'dominant_emotion': img_dominant_emotion,
            'emotion_scores': img_emotion_scores,
            'recommendations': img_recommendations,
            'analysis': image_analysis
        }
        
        logger.info(f"Image-detected emotion: {img_dominant_emotion}")
    except Exception as e:
        logger.error(f"Error in image-based recommendation: {e}")
        logger.exception("Full traceback:")
        results['image'] = None
    
    # 2. Slider + Text based recommendations
    if emotion_data and 'emotions' in emotion_data and emotion_data['emotions']:
        logger.info(f"Generating slider-based recommendations for {image_name}")
        try:
            slider_emotions = emotion_data['emotions']
            slider_dominant_emotion = max(slider_emotions.items(), key=lambda x: x[1])[0]
            
            # Convert emotion_data to vector for recommendation
            emotion_vector = np.zeros(len(emosi.image_detector.emotion_categories))
            for i, emotion in enumerate(emosi.image_detector.emotion_categories):
                emotion_vector[i] = slider_emotions.get(emotion, 0.0)
            
            # Normalize the vector
            norm = np.linalg.norm(emotion_vector)
            if norm > 0:
                emotion_vector = emotion_vector / norm
            
            # Get recommendations based on slider data
            slider_recommendations = emosi.recommend_by_emotion(
                emotion_vector=emotion_vector,
                num_recommendations=num_recommendations
            )
            
            # Get text description if available
            description = emotion_data.get('description')
            
            # If we have a text description, enhance with text-based recommendations
            text_recommendations = []
            if description:
                logger.info(f"Enhancing with text description: {description}")
                text_recommendations = emosi.recommend_by_text(
                    query_text=description,
                    num_recommendations=num_recommendations,
                    year_cutoff=year_cutoff
                )
            
            results['slider_text'] = {
                'dominant_emotion': slider_dominant_emotion,
                'emotion_scores': slider_emotions,
                'recommendations': slider_recommendations,
                'description': description,
                'text_recommendations': text_recommendations
            }
            
            logger.info(f"Slider-based dominant emotion: {slider_dominant_emotion}")
        except Exception as e:
            logger.error(f"Error in slider-based recommendation: {e}")
            logger.exception("Full traceback:")
            results['slider_text'] = None
    else:
        logger.info(f"No slider data found for {image_name}. Skipping slider-based recommendations.")
        results['slider_text'] = None
    
    # 3. Combined recommendations (image + slider + text)
    if results['image'] and results['slider_text']:
        logger.info(f"Generating combined recommendations for {image_name}")
        try:
            # Create a combined emotion vector by averaging image and slider vectors
            img_vector = np.zeros(len(emosi.image_detector.emotion_categories))
            for i, emotion in enumerate(emosi.image_detector.emotion_categories):
                img_vector[i] = results['image']['emotion_scores'].get(emotion, 0.0)
            
            slider_vector = np.zeros(len(emosi.image_detector.emotion_categories))
            for i, emotion in enumerate(emosi.image_detector.emotion_categories):
                slider_vector[i] = results['slider_text']['emotion_scores'].get(emotion, 0.0)
            
            # Normalize vectors
            img_norm = np.linalg.norm(img_vector)
            if img_norm > 0:
                img_vector = img_vector / img_norm
                
            slider_norm = np.linalg.norm(slider_vector)
            if slider_norm > 0:
                slider_vector = slider_vector / slider_norm
            
            # Combine with weight (60% slider, 40% image)
            combined_vector = 0.6 * slider_vector + 0.4 * img_vector
            
            # Normalize combined vector
            combined_norm = np.linalg.norm(combined_vector)
            if combined_norm > 0:
                combined_vector = combined_vector / combined_norm
            
            # Get the dominant emotion from the combined vector
            dominant_idx = np.argmax(combined_vector)
            combined_dominant_emotion = emosi.image_detector.emotion_categories[dominant_idx]
            
            # Create combined emotion scores
            combined_scores = {}
            for i, emotion in enumerate(emosi.image_detector.emotion_categories):
                if combined_vector[i] > 0:
                    combined_scores[emotion] = float(combined_vector[i])
            
            # Get recommendations based on combined vector
            combined_recommendations = emosi.recommend_by_emotion(
                emotion_vector=combined_vector,
                num_recommendations=num_recommendations
            )
            
            # Calculate emotional similarity between image and slider
            similarity = np.dot(img_vector, slider_vector)
            similarity = (similarity + 1) / 2  # Normalize to 0-1 range
            
            results['combined'] = {
                'dominant_emotion': combined_dominant_emotion,
                'emotion_scores': combined_scores,
                'recommendations': combined_recommendations,
                'similarity': float(similarity)
            }
            
            logger.info(f"Combined dominant emotion: {combined_dominant_emotion}")
            logger.info(f"Image-slider similarity: {similarity:.2f}")
        except Exception as e:
            logger.error(f"Error in combined recommendation: {e}")
            logger.exception("Full traceback:")
            results['combined'] = None
    else:
        logger.info(f"Cannot generate combined recommendations for {image_name} (missing image or slider data)")
        results['combined'] = None
    
    return results

def analyze_images_and_generate_recommendations(input_folder, csv_file_path, output_dir, num_recommendations=10, year_cutoff=2015):
    """
    Analyze all images in the input folder and generate music recommendations using multiple modalities.
    Store results in output files.
    """
    # Import modules needed inside this function
    import os
    import time
    import torch
    
    logger = setup_logging(output_dir)
    
    # Process CSV emotion data
    logger.info(f"Processing emotion data from CSV: {csv_file_path}")
    image_emotions, name_emotions, clean_name_emotions = process_csv_emotion_data(csv_file_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary CSV file
    summary_file = os.path.join(output_dir, "recommendation_summary.csv")
    with open(summary_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Image File', 
            'Person Name',
            'Image Emotion', 
            'Slider Emotion',
            'Combined Emotion',
            'Emotion Similarity',
            'Person Description',
            'Top Image Recommendation',
            'Top Slider Recommendation',
            'Top Combined Recommendation'
        ])
        
        # Get all image files
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.heic'))]
        
        # Process images one at a time (reduced batch size from 2 to 1)
        batch_size = 1  # Process 1 image at a time to manage memory
        for batch_idx in range(0, len(image_files), batch_size):
            try:
                # Set specific memory limits for the model to avoid OOM errors
                import torch
                import os
                
                # Set memory limits before model initialization
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Initialize EMOSI facade with lower precision to reduce memory usage
                emosi = EmosiFacade(use_dummy=False, model_precision="float16")  
                
                # Process a batch of images
                batch_files = image_files[batch_idx:batch_idx + batch_size]
                for file_name in batch_files:
                    image_path = os.path.join(input_folder, file_name)
                    
                    logger.info(f"\n{'='*80}\nProcessing image: {file_name}\n{'='*80}")
                    
                    # Resize the image before processing to reduce memory usage
                    try:
                        max_size = (512, 512)  # Limit image size
                        with Image.open(image_path) as img:
                            img.thumbnail(max_size)
                            temp_path = os.path.join(output_dir, f"temp_{file_name}")
                            img.save(temp_path)
                            image_path = temp_path
                            logger.info(f"Resized image to maximum dimensions of {max_size}")
                    except Exception as e:
                        logger.warning(f"Could not resize image: {e}")
                    
                    # Find matching emotion data
                    emotion_data = find_matching_emotion_data(file_name, image_emotions, name_emotions, clean_name_emotions)
                    
                    if emotion_data:
                        person_name = emotion_data.get('name', 'Unknown')
                        logger.info(f"Found emotion data for {file_name} from person: {person_name}")
                    else:
                        person_name = "Unknown"
                        logger.info(f"No matching slider data found for {file_name}")
                    
                    # Generate all recommendations
                    results = analyze_and_recommend(
                        emosi=emosi,
                        image_path=image_path,
                        emotion_data=emotion_data,
                        num_recommendations=num_recommendations,
                        year_cutoff=year_cutoff,
                        logger=logger
                    )
                    
                    # Create detailed output file for this image
                    output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_recommendations.txt")
                    with open(output_file, 'w') as f:
                        f.write(f"Image: {file_name}\n")
                        f.write(f"Person: {person_name}\n\n")
                        
                        # Include full image analysis
                        if results['image'] and results['image']['analysis']:
                            f.write("FULL IMAGE ANALYSIS\n")
                            f.write("="*80 + "\n")
                            f.write(f"{results['image']['analysis']}\n\n")
                        
                        # Include person's description if available
                        if results['slider_text'] and results['slider_text']['description']:
                            f.write("PERSON'S DESCRIPTION\n")
                            f.write("="*80 + "\n")
                            f.write(f"{results['slider_text']['description']}\n\n")
                        
                        # 1. IMAGE-BASED SECTION
                        f.write("1. IMAGE-BASED RECOMMENDATIONS\n")
                        f.write("="*80 + "\n")
                        if results['image']:
                            f.write(f"Detected Emotion: {results['image']['dominant_emotion']}\n\n")
                            f.write("Emotion Profile:\n")
                            for emotion, score in sorted(results['image']['emotion_scores'].items(), key=lambda x: x[1], reverse=True):
                                if score > 0.01:  # Only show significant emotions
                                    f.write(f"  {emotion}: {score:.3f}\n")
                            
                            f.write("\nRecommended Songs:\n")
                            f.write(format_recommendation_output(results['image']['recommendations']))
                        else:
                            f.write("Image-based analysis failed.\n")
                        
                        # 2. SLIDER + TEXT SECTION
                        f.write("\n\n2. SLIDER + TEXT BASED RECOMMENDATIONS\n")
                        f.write("="*80 + "\n")
                        if results['slider_text']:
                            f.write(f"Dominant Emotion: {results['slider_text']['dominant_emotion']}\n\n")
                            f.write("Emotion Profile from Sliders:\n")
                            for emotion, score in sorted(results['slider_text']['emotion_scores'].items(), key=lambda x: x[1], reverse=True):
                                if score > 0.01:  # Only show significant emotions
                                    f.write(f"  {emotion}: {score:.3f}\n")
                            
                            f.write("\nSlider-Based Recommended Songs:\n")
                            f.write(format_recommendation_output(results['slider_text']['recommendations']))
                            
                            # Include text-based recommendations if available
                            if results['slider_text']['text_recommendations']:
                                f.write("\nText-Based Recommended Songs (based on description):\n")
                                f.write(format_recommendation_output(results['slider_text']['text_recommendations']))
                        else:
                            f.write("No slider data available for this image.\n")
                        
                        # 3. COMBINED SECTION
                        f.write("\n\n3. COMBINED RECOMMENDATIONS (IMAGE + SLIDER + TEXT)\n")
                        f.write("="*80 + "\n")
                        if results['combined']:
                            f.write(f"Combined Dominant Emotion: {results['combined']['dominant_emotion']}\n")
                            f.write(f"Image-Slider Emotion Similarity: {results['combined']['similarity']:.2f}\n\n")
                            
                            f.write("Combined Emotion Profile:\n")
                            for emotion, score in sorted(results['combined']['emotion_scores'].items(), key=lambda x: x[1], reverse=True):
                                if score > 0.01:  # Only show significant emotions
                                    f.write(f"  {emotion}: {score:.3f}\n")
                            
                            f.write("\nCombined Recommended Songs:\n")
                            f.write(format_recommendation_output(results['combined']['recommendations']))
                        else:
                            f.write("Combined analysis not available (requires both image and slider data).\n")
                    
                    # Add to summary CSV
                    img_recommendation = "N/A"
                    slider_recommendation = "N/A"
                    combined_recommendation = "N/A"
                    
                    if results['image'] and results['image']['recommendations']:
                        img_recommendation = results['image']['recommendations'][0].get('name', 
                                           results['image']['recommendations'][0].get('title', 'Unknown'))
                    
                    if results['slider_text'] and results['slider_text']['recommendations']:
                        slider_recommendation = results['slider_text']['recommendations'][0].get('name',
                                              results['slider_text']['recommendations'][0].get('title', 'Unknown'))
                    
                    if results['combined'] and results['combined']['recommendations']:
                        combined_recommendation = results['combined']['recommendations'][0].get('name',
                                               results['combined']['recommendations'][0].get('title', 'Unknown'))
                    
                    description = "N/A"
                    if results['slider_text'] and results['slider_text']['description']:
                        description = results['slider_text']['description']
                    
                    similarity = "N/A"
                    if results['combined']:
                        similarity = f"{results['combined']['similarity']:.2f}"
                    
                    csv_writer.writerow([
                        file_name,
                        person_name,
                        results['image']['dominant_emotion'] if results['image'] else "N/A",
                        results['slider_text']['dominant_emotion'] if results['slider_text'] else "N/A",
                        results['combined']['dominant_emotion'] if results['combined'] else "N/A",
                        similarity,
                        description,
                        img_recommendation,
                        slider_recommendation,
                        combined_recommendation
                    ])
                
                # Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after batch processing")
                
                # Free the EMOSI object to release GPU memory
                del emosi
                
                # Small delay to allow GPU to recover
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                logger.exception("Full traceback:")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Analysis complete. Results saved in {output_dir}")
    logger.info(f"{'='*80}\n")

def run_cli_demo():
    """Run the command line interface demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EMOSI: Emotion-based Music Selection Interface')
    parser.add_argument('--text', type=str, help='Text query for music recommendation')
    parser.add_argument('--mode', type=str, default='text', choices=['image', 'text', 'combined', 'batch'], 
                      help='Mode to run: image-based, text-based, combined, or batch processing')
    parser.add_argument('--year-cutoff', type=int, default=2015, 
                      help='Prefer songs released after this year (default: 2015)')
    parser.add_argument('--num-recommendations', type=int, default=10,
                      help='Number of recommendations to return (default: 10)')
    parser.add_argument('--image', type=str, help='Path to image for emotion-based recommendation')
    parser.add_argument('--input-folder', type=str, default='/home/ubuntu/inputs',
                      help='Folder containing images to analyze in batch mode')
    parser.add_argument('--csv-file', type=str, default='/home/ubuntu/emosi_survey_responses_apr23.csv',
                      help='CSV file with slider-based emotion data')
    parser.add_argument('--output-dir', type=str, default='./emosi_output',
                      help='Directory to store output files and logs')
    
    args = parser.parse_args()
    
    # Initialize the EMOSI system with dummy mode for testing
    print("\n" + "="*80)
    print("EMOSI: Emotion-based Music Selection Interface")
    print("="*80)
    
    try:
        if args.mode == 'batch':
            print(f"Running batch analysis on images in {args.input_folder}")
            print(f"Using slider data from {args.csv_file}")
            print(f"Results will be saved to {args.output_dir}")
            
            analyze_images_and_generate_recommendations(
                input_folder=args.input_folder,
                csv_file_path=args.csv_file,
                output_dir=args.output_dir,
                num_recommendations=args.num_recommendations,
                year_cutoff=args.year_cutoff
            )
        elif args.mode == 'text' and args.text:
            print(f"Running text query: '{args.text}' with year cutoff: {args.year_cutoff}")
            
            emosi = EmosiFacade(use_dummy=False)
            
            recommendations = emosi.recommend_by_text(
                query_text=args.text, 
                num_recommendations=args.num_recommendations,
                year_cutoff=args.year_cutoff
            )
            
            print("\nRecommended songs:")
            print(format_recommendation_output(recommendations))
        elif args.mode == 'image' and args.image:
            print(f"Running image-based recommendation using: '{args.image}'")
            
            emosi = EmosiFacade(use_dummy=False)
            
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
            print("  For batch mode: --mode=batch [--input-folder=/path/to/images] [--csv-file=/path/to/csv]")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("Thank you for using EMOSI!")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_cli_demo()
