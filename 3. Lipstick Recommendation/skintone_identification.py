import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ExifTags, ImageOps

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def simple_gray_world_balance(img):
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
    arr = np.array(rgb_values)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    mask = (np.abs(arr - mean) < z_threshold * std).all(axis=1)
    return arr[mask]

def get_skintone_id(image_path, excel_path='./recommendation_datasets/Mapping table 2 Sephora.xlsx', return_intermediates=False):
    mp_face_mesh = mp.solutions.face_mesh

    # Original image
    image_pil = Image.open(image_path)
    image_pil = ImageOps.exif_transpose(image_pil)  # Fix orientation if needed
    exif = None
    try:
        exif = image_pil.getexif()
        print(exif)
    except:
        pass
    
    make_value = None
    if exif is not None:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            print(tag, tag_name, value)
            if tag_name=='Make':
                print(tag_name, value)
                make_value = value
                break

    if make_value is not None:
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Gray world balance
        grayworld_img = simple_gray_world_balance(image.copy())
        # Gamma correction
        gamma_img = gamma_correction(grayworld_img.copy(), gamma=1.1)
    else:
        # Upload image likely to be non-real-life images
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        grayworld_img = image.copy()
        gamma_img = image.copy()

    # Final RGB image for detection
    rgb_image = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_image.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            landmark_indices = {
                "Forehead": [69, 108, 151, 337, 299, 109, 10, 338, 297, 67, 9],
                "Chin": [32, 194, 199, 208, 428, 201, 200, 219, 211, 421, 418, 424, 416, 432],
                "left_cheek_landmarks": [50, 205, 250],
                "right_cheek_landmarks": [280, 330, 425],
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

            if top_10_rgb_values:
                top_10_rgb_values = remove_outliers(top_10_rgb_values, z_threshold=2.0)
                if len(top_10_rgb_values) == 0:
                    raise Exception("No valid skin pixels after outlier removal.")
                
                average_rgb = np.mean(top_10_rgb_values, axis=0).astype(np.uint8)
                avg_lab = cv2.cvtColor(average_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB)
                final_avg_rgb = cv2.cvtColor(avg_lab, cv2.COLOR_LAB2RGB)[0,0,:]

                skin_tone_df = pd.read_excel(excel_path, sheet_name='skinID to skinID_Sephora')
                skin_tone_rgb_values = skin_tone_df[['R', 'G', 'B']].values
                distances = np.linalg.norm(skin_tone_rgb_values - final_avg_rgb, axis=1)
                closest_index = np.argmin(distances)
                closest_skin_tone = skin_tone_df.iloc[closest_index]

                skin_id = closest_skin_tone['Skin_ID']
                skin_id_sephora = closest_skin_tone['Skin_ID_Sephora']

                if return_intermediates:
                    # Convert intermediate numpy arrays back to PIL for display
                    grayworld_img_pil = Image.fromarray(cv2.cvtColor(grayworld_img, cv2.COLOR_BGR2RGB))
                    gamma_img_pil = Image.fromarray(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
                    final_rgb_img_pil = Image.fromarray(rgb_image)
                    return skin_id, skin_id_sephora, grayworld_img_pil, gamma_img_pil, final_rgb_img_pil
                else:
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
