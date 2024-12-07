import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageEnhance, ExifTags

def gamma_correction(img, gamma=1.0):
    # Build a lookup table mapping pixel values [0..255] to gamma-corrected values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def simple_gray_world_balance(img):
    # Simple gray world white balance
    img = img.astype(np.float32)
    mean_b = np.mean(img[..., 0])
    mean_g = np.mean(img[..., 1])
    mean_r = np.mean(img[..., 2])
    
    avg_gray = (mean_b + mean_g + mean_r) / 3.0
    scale_b = avg_gray / mean_b
    scale_g = avg_gray / mean_g
    scale_r = avg_gray / mean_r
    
    img[..., 0] *= scale_b
    img[..., 1] *= scale_g
    img[..., 2] *= scale_r
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def remove_outliers(rgb_values, z_threshold=2.0):
    # Remove outliers based on a simple Z-score method
    arr = np.array(rgb_values)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    # Keep only those within z_threshold in all channels
    mask = (np.abs(arr - mean) < z_threshold * std).all(axis=1)
    return arr[mask]

def get_brightness_factor(iso):
    # Define piecewise linear brightness factor based on ISO
    upscale=1.05
    downscale=0.95
    
    if iso <= 50:
        return downscale
    elif 50 < iso < 150:
        # Linear between 50->0.5 and 150->1.0
        slope = (1.0 - downscale) / (150 - 50)
        return downscale + (iso - 50) * slope
    elif iso == 150:
        return 1.0
    elif 150 < iso < 400:
        # Linear between 150->1.0 and 400->1.5
        slope = (upscale - 1.0) / (400 - 150)
        return 1.0 + (iso - 150) * slope
    else: # iso >= 400
        return upscale

def get_skintone_id(image_path, excel_path='./recommendation_datasets/Mapping table 2 Sephora.xlsx'):
    mp_face_mesh = mp.solutions.face_mesh
    
    # --- Retrieve ISO and adjust brightness ---
    image_pil = Image.open(image_path)
    exif = None
    if hasattr(image_pil, '_getexif'):
        exif = image_pil._getexif()
    
    iso_value = 150  # default ISO if not found
    if exif is not None:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'ISOSpeedRatings':
                iso_value = value
                break
    
    # Compute brightness factor
    factor = get_brightness_factor(iso_value)

    # Adjust brightness using PIL
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(factor)

    # Convert the PIL image to OpenCV (BGR) format
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        # Step 1: Apply white balance
        image = simple_gray_world_balance(image)
        
        # Adjust gamma as needed; 1.0 means no correction, adjust if needed.
        image = gamma_correction(image, gamma=1.1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            landmark_indices = {
                "Forehead": [69, 108, 151, 337, 299, 109, 10, 338, 297, 67, 9],
                "Chin": [32, 194, 199, 208, 428, 201, 200, 219, 211, 421, 418, 424, 416, 432],
                "left_cheek_landmarks": [50, 205, 250],
                "right_cheek_landmarks":[280, 330, 425],
            }
            
            luminance_values = []
            for region, indices in landmark_indices.items():
                for idx in indices:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    rgb = rgb_image[y, x]
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    luminance_values.append((luminance, rgb))

            # Sort and take top bright pixels
            luminance_values.sort(reverse=True, key=lambda x: x[0])
            top_10_rgb_values = [rgb for _, rgb in luminance_values[:10]]
            
            if top_10_rgb_values:
                # Remove outliers
                top_10_rgb_values = remove_outliers(top_10_rgb_values, z_threshold=2.0)
                if len(top_10_rgb_values) == 0:
                    raise Exception("No valid skin pixels after outlier removal.")
                
                # Average the resulting RGB values
                average_rgb = np.mean(top_10_rgb_values, axis=0).astype(np.uint8)
                
                # Convert avg to LAB then back to RGB
                avg_lab = cv2.cvtColor(average_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB)
                final_avg_rgb = cv2.cvtColor(avg_lab, cv2.COLOR_LAB2RGB)[0,0,:]
                
                # Match against reference skin tones
                skin_tone_df = pd.read_excel(excel_path, sheet_name='skinID to skinID_Sephora')
                skin_tone_rgb_values = skin_tone_df[['R', 'G', 'B']].values
                distances = np.linalg.norm(skin_tone_rgb_values - final_avg_rgb, axis=1)
                closest_index = np.argmin(distances)
                closest_skin_tone = skin_tone_df.iloc[closest_index]
                
                skin_id = closest_skin_tone['Skin_ID']
                skin_id_sephora = closest_skin_tone['Skin_ID_Sephora']
                return skin_id, skin_id_sephora
            else:
                raise Exception("No valid skin tone detected.")
        else:
            raise Exception("No face detected in the image.")

# Test the function if running this file directly
if __name__ == "__main__":
    try:
        # Test with a sample image
        skin_tone_id, skin_id_sephora = get_skintone_id("test_image.jpg")
        print(f"Expert Skin Tone ID: {skin_tone_id}")
        print(f"Sephora Skin ID: {skin_id_sephora}")
    except Exception as e:
        print(f"Error: {str(e)}")
