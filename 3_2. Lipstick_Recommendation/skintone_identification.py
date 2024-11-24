import random
from PIL import Image
import pandas as pd

def load_skin_tone_ids():
    """
    Load and extract unique skin tone IDs from the recommendation files
    """
    try:
        # Load the Excel files
        expert_df = pd.read_excel("expertRecommendation_top_10_skus_per_RBG.xlsx")
        review_df = pd.read_excel("skinToneRecommendation_top_20_skus_per_skinTone.xlsx")
        
        # Get unique skin tone IDs
        expert_skin_ids = sorted(expert_df['Skin_ID'].unique())
        sephora_skin_ids = sorted(review_df['Skin_ID_Sephora'].unique())
        
        return expert_skin_ids, sephora_skin_ids
        
    except Exception as e:
        raise Exception(f"Error loading skin tone IDs: {str(e)}")

def get_skintone_id(image_path):
    """
    Takes an image path and returns randomly generated skin tone IDs from the available options.
    
    Args:
        image_path (str): Path to the uploaded image
        
    Returns:
        tuple: (skin_tone_id, skin_id_sephora)
    """
    try:
        # Load and verify the image
        image = Image.open(image_path)
        
        # Get available skin tone IDs from the files
        expert_skin_ids, sephora_skin_ids = load_skin_tone_ids()
        
        # Randomly select one ID from each list
        skin_tone_id = random.choice(expert_skin_ids)
        skin_id_sephora = random.choice(sephora_skin_ids)
        
        return skin_tone_id, skin_id_sephora
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# Test the function if running this file directly
if __name__ == "__main__":
    try:
        # Test with a sample image
        skin_tone_id, skin_id_sephora = get_skintone_id("test_image.jpg")
        print(f"Expert Skin Tone ID: {skin_tone_id}")
        print(f"Sephora Skin ID: {skin_id_sephora}")
    except Exception as e:
        print(f"Error: {str(e)}")