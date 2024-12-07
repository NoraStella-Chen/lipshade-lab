from PIL import Image
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np

def get_skintone_id(image_path, excel_path='./recommendation_datasets/Mapping table 2 Sephora.xlsx'):
    """
    Detect skin tone from image and map to Sephora's ID using color distance.
    
    Args:
        image_path (str): Path to the input image.
        excel_path (str): Path to the Excel file with skin tone mappings.
    
    Returns:
        tuple: (skin_id, skin_id_sephora) - IDs for the detected skin tone.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    
    image = cv2.imread(image_path)
    print('success image read')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_image.shape
    print('image height',h,'image width',w)
    
    results = face_mesh.process(rgb_image)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmark_indices = {
            "Forehead": [69, 108, 151, 337, 299, 109, 10, 338, 297, 67, 9],
            "Chin": [32, 194, 199, 208, 428, 201, 200, 219, 211, 421, 418, 424, 416, 432]
        }
        
        luminance_values = []
        
        for region, indices in landmark_indices.items():
            for idx in indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                rgb = rgb_image[y, x]
                luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                luminance_values.append((luminance, rgb))
        
        luminance_values.sort(reverse=True, key=lambda x: x[0])
        top_10_rgb_values = [rgb for _, rgb in luminance_values[:10]]
        print('Top N:',top_10_rgb_values)
        
        if top_10_rgb_values:
            top_10_rgb_values = np.array(top_10_rgb_values)
            average_rgb = np.mean(top_10_rgb_values, axis=0).astype(int)
            print(average_rgb)
            skin_tone_df = pd.read_excel(excel_path, sheet_name='skinID to skinID_Sephora')
            skin_tone_rgb_values = skin_tone_df[['R', 'G', 'B']].values
            distances = np.linalg.norm(skin_tone_rgb_values - average_rgb, axis=1)
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