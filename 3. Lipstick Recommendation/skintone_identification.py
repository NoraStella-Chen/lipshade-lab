import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ExifTags, ImageOps

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0)**inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
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

def load_and_preprocess_image(image_path):
    image_pil = Image.open(image_path)
    # Fix orientation if needed
    image_pil = ImageOps.exif_transpose(image_pil)
    return image_pil

def extract_make_exif(image_pil):
    exif = None
    make_value = None
    try:
        exif = image_pil.getexif()
    except:
        pass

    if exif is not None:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == 'Make':
                make_value = value
                break
    return make_value

def balance_and_correct_image(image_bgr, make_value):
    if make_value is not None:
        grayworld_img = simple_gray_world_balance(image_bgr.copy())
        gamma_img = gamma_correction(grayworld_img.copy(), gamma=1.1)
    else:
        # For non-real images, skip corrections
        grayworld_img = image_bgr.copy()
        gamma_img = image_bgr.copy()

    return grayworld_img, gamma_img

def detect_face_landmarks(rgb_image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
    return results

def extract_skin_rgb(results, rgb_image):
    if not results.multi_face_landmarks:
        raise Exception("No face detected in the image.")

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = rgb_image.shape

    # Landmark indices as before
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
    return top_10_rgb_values

def average_skin_color(rgb_values):
    if not rgb_values:
        raise Exception("No valid skin tone detected.")

    rgb_values = remove_outliers(rgb_values, z_threshold=2.0)
    if len(rgb_values) == 0:
        raise Exception("No valid skin pixels after outlier removal.")

    average_rgb = np.mean(rgb_values, axis=0).astype(np.uint8)
    avg_lab = cv2.cvtColor(average_rgb.reshape(1,1,3), cv2.COLOR_RGB2LAB)
    final_avg_rgb = cv2.cvtColor(avg_lab, cv2.COLOR_LAB2RGB)[0,0,:]

    return final_avg_rgb

def match_skin_tone(final_avg_rgb, excel_path):
    skin_tone_df = pd.read_excel(excel_path, sheet_name='skinID to skinID_Sephora')
    skin_tone_rgb_values = skin_tone_df[['R', 'G', 'B']].values
    distances = np.linalg.norm(skin_tone_rgb_values - final_avg_rgb, axis=1)
    closest_index = np.argmin(distances)
    closest_skin_tone = skin_tone_df.iloc[closest_index]

    skin_id = closest_skin_tone['Skin_ID']
    skin_id_sephora = closest_skin_tone['Skin_ID_Sephora']
    return skin_id, skin_id_sephora

def get_skintone_id(image_path, excel_path='./recommendation_datasets/Mapping table 2 Sephora.xlsx', return_intermediates=False):
    # Load and preprocess image
    image_pil = load_and_preprocess_image(image_path)
    make_value = extract_make_exif(image_pil)
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Balance and gamma correct based on EXIF
    grayworld_img, gamma_img = balance_and_correct_image(image_bgr, make_value)

    rgb_image = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    results = detect_face_landmarks(rgb_image)

    # Extract top skin pixels
    top_10_rgb_values = extract_skin_rgb(results, rgb_image)
    final_avg_rgb = average_skin_color(top_10_rgb_values)

    # Match against skin tone database
    skin_id, skin_id_sephora = match_skin_tone(final_avg_rgb, excel_path)

    if return_intermediates:
        # Create color patch of the final average RGB
        color_square = np.full((300, 300, 3), final_avg_rgb, dtype=np.uint8)
        color_square_pil = Image.fromarray(color_square)

        return skin_id, skin_id_sephora, color_square_pil
    else:
        return skin_id, skin_id_sephora

# Example Test 
if __name__ == "__main__":
    try:
        # Test with a sample image
        skin_tone_id, skin_id_sephora = get_skintone_id("test_image.jpg")
        print(f"Expert Skin Tone ID: {skin_tone_id}")
        print(f"Sephora Skin ID: {skin_id_sephora}")
    except Exception as e:
        print(f"Error: {str(e)}")
