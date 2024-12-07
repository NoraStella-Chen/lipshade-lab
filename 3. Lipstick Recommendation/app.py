import streamlit as st
import pandas as pd
import numpy as np
import os
from skintone_identification import get_skintone_id
import plotly.express as px
from PIL import Image, ImageOps
import cv2

st.set_page_config(layout="wide")

# Load data files
file_path_expert_skintone = "recommendation_datasets/expertRecommendation_top_10_skus_per_RBG_w_review_summarized.xlsx"
file_path_review_skintone = "recommendation_datasets/skinToneRecommendation_top_20_skus_per_skinTone_w_review_summarized.xlsx"
file_path_colorcluster = "recommendation_datasets/colorClusterRecommendation_top_20_skus_per_colorCluster_w_review_summarized.xlsx"

expert_df = pd.read_excel(file_path_expert_skintone)
review_df = pd.read_excel(file_path_review_skintone)
color_df = pd.read_excel(file_path_colorcluster)

# Helper Functions
def create_id_name_mapping(df, id_column, name_column):
    return dict(zip(df[id_column], df[name_column]))

def format_selection_options(mapping):
    return {f"{name} (ID: {id})": id for id, name in mapping.items()}

def rgb_to_hex(r, g, b):
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def load_and_process_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)  # Fix orientation if needed
    
    max_width = 300
    aspect_ratio = image.width / image.height
    new_width = min(max_width, image.width)
    new_height = int(new_width / aspect_ratio)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def process_uploaded_image():
    uploaded_file = st.file_uploader(
        "Choose an image to identify your skin tone", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image (original)
        processed_image = load_and_process_image(uploaded_file)
        st.image(processed_image, caption="Uploaded Image")

        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Modified get_skintone_id to return intermediate images as well
            skin_tone_id, skin_id_sephora, grayworld_img, gamma_img, final_rgb_img = get_skintone_id("temp_image.jpg", return_intermediates=True)

            # Show images side by side
            st.markdown("### Processing Steps")
            col1, col2, col3, col4 = st.columns(4)
            col1.image(processed_image, caption="Original (Uploaded)")
            col2.image(grayworld_img, caption="Gray World Balanced")
            col3.image(gamma_img, caption="Gamma Corrected")
            col4.image(final_rgb_img, caption="Final RGB Image")

            # Get mappings
            expert_mapping = create_id_name_mapping(expert_df, "Skin_ID", "Skin_Name")
            sephora_mapping = create_id_name_mapping(review_df, "Skin_ID_Sephora", "Skin_Tone_Name")

            expert_name = expert_mapping.get(skin_tone_id, f"Unknown (ID: {skin_tone_id})")
            sephora_name = sephora_mapping.get(skin_id_sephora, f"Unknown (ID: {skin_id_sephora})")

            with st.container():
                st.markdown("### Detected Skin Tones")
                st.success(f"Expert System: {expert_name}")
                st.success(f"Sephora System: {sephora_name}")

            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")

            return skin_tone_id, skin_id_sephora

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None

    return None, None

def get_color_options(df):
    color_options = []
    for _, row in df.drop_duplicates(["ColorCluster", "ClusterColorDescription"]).iterrows():
        color_hex = rgb_to_hex(row["cluster_R"], row["cluster_G"], row["cluster_B"])
        color_options.append({
            "name": row["ClusterColorDescription"],
            "id": row["ColorCluster"],
            "color": color_hex,
        })
    return color_options

def get_recommendations(
    df, filter_column, filter_value, n_recommendations=5, source_name="Recommendation"
):
    recommendations = df[df[filter_column] == filter_value]
    recommendations["weighted_rating"] = (
        recommendations["skuRating"] * recommendations["skuTotalReviews"]
    )
    recommendations = recommendations.sort_values("weighted_rating", ascending=False).head(n_recommendations)
    recommendations["Source"] = source_name
    return recommendations

def get_expert_recommendations(skin_tone_id, n_recommendations=5):
    return get_recommendations(
        expert_df, "Skin_ID", skin_tone_id, n_recommendations, "Expert Choice"
    )

def get_review_recommendations(skin_id_sephora, n_recommendations=5):
    return get_recommendations(
        review_df,
        "Skin_ID_Sephora",
        skin_id_sephora,
        n_recommendations,
        "Top Rated by Users",
    )

def get_color_recommendations(color_cluster_id, n_recommendations=5):
    return get_recommendations(
        color_df, "ColorCluster", color_cluster_id, n_recommendations, "Color Match"
    )

def extract_first_price(price_str):
    try:
        cleaned = str(price_str).replace("$", "").strip()
        if " - " in cleaned:
            return float(cleaned.split(" - ")[0])
        return float(cleaned)
    except (ValueError, AttributeError):
        return None

def format_dataframe(df):
    def format_price(x):
        if pd.notnull(x):
            if not str(x).startswith("$"):
                return f"${str(x)}"
            return str(x)
        return ""

    def format_rating(x):
        try:
            if pd.notnull(x):
                return f"{float(x):.1f}/5"
            return ""
        except (ValueError, TypeError):
            return str(x)

    df["Product Info"] = (
        "Brand: "
        + df["brandName"].astype(str)
        + "\nName:"
        + df["displayName"].astype(str)
        + "\nPrice: "
        + df["currentSku_listPrice"].apply(format_price).astype(str)
        + "\nRating: "
        + df["skuRating"].apply(format_rating).astype(str)
        + " ("
        + df["skuTotalReviews"].astype(str)
        + " reviews)"
    )
    formatted_df = df[["skuID", "Product Info", "Sentiment", "lipstick_image_base64", "URL"]].copy()

    formatted_df["URL"] = formatted_df["URL"].apply(
        lambda x: (f'<a href="{x}" target="_blank">View Product</a>' if pd.notnull(x) else "")
    )

    def convert_base64_to_img(base64_str, width="50px"):
        if base64_str and pd.isnull(base64_str) == False and len(str(base64_str)) > 0:
            return f'<img src="data:image/png;base64,{base64_str}" style="width: {width}"/>'
        else:
            return "No cover"

    formatted_df["lipstick_image_base64"] = formatted_df["lipstick_image_base64"].apply(
        lambda x: convert_base64_to_img(x) if x else ""
    )

    def convert_text_to_scrollable_html(text, max_height="100px"):
        cleaned_text = text.lstrip('\n').rstrip('\n').replace('\n\n','\n').replace('\n\n','\n').replace('\n \n','\n').replace('• ','').replace('* ','').replace('- ','').replace('\n', '<br>')
        return f'''<div style="max-height: {max_height}; overflow-y: auto; white-space: pre-wrap;">{cleaned_text}</div>'''

    formatted_df["Sentiment"] = formatted_df["Sentiment"].apply(
        lambda x: convert_text_to_scrollable_html(x) if x else ""
    )
    formatted_df["Product Info"] = formatted_df["Product Info"].apply(
        lambda x: convert_text_to_scrollable_html(x) if x else ""
    )

    column_rename = {
        "skuID": "SKU",
        "Product Info": "Product Info",
        "Sentiment": "Review Summary by LLM",
        "lipstick_image_base64": "SKU Image",
        "URL": "Link (USA only)",
    }
    formatted_df.rename(columns=column_rename, inplace=True)

    return formatted_df

def add_filters(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        brand_filter = st.multiselect("Filter by Brand", df["brandName"].unique())
    with col2:
        price_values = df["currentSku_listPrice"].apply(extract_first_price)
        valid_prices = price_values.dropna()
        if not valid_prices.empty:
            min_price = st.number_input(
                "Min Price",
                min_value=float(valid_prices.min()),
                max_value=float(valid_prices.max()),
                value=float(valid_prices.min()),
            )
            max_price = st.number_input(
                "Max Price",
                min_value=min_price,
                max_value=float(valid_prices.max()),
                value=float(valid_prices.max()),
            )
        else:
            min_price = 0
            max_price = float("inf")
    with col3:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0)

    filtered_df = df.copy()
    if brand_filter:
        filtered_df = filtered_df[filtered_df["brandName"].isin(brand_filter)]

    filtered_df = filtered_df[
        (filtered_df["currentSku_listPrice"].apply(extract_first_price) >= min_price)
        & (filtered_df["currentSku_listPrice"].apply(extract_first_price) <= max_price)
        & (filtered_df["skuRating"] >= min_rating)
    ]
    return filtered_df

def add_visualizations(df):
    st.subheader("Product Analysis")
    df["Price_Numeric"] = df["currentSku_listPrice"].apply(extract_first_price)

    fig_price = px.histogram(
        df,
        x="Price_Numeric",
        title="Price Distribution",
        labels={"Price_Numeric": "Price ($)"},
    )
    st.plotly_chart(fig_price)

    fig_scatter = px.scatter(
        df,
        x="Price_Numeric",
        y="skuRating",
        color="brandName",
        title="Price vs Rating by Brand",
        labels={"Price_Numeric": "Price ($)", "skuRating": "Rating"},
    )
    st.plotly_chart(fig_scatter)

def add_export_option(df):
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download recommendations as CSV",
            data=csv,
            file_name="lipstick_recommendations.csv",
            mime="text/csv",
        )

# Main Layout
st.markdown(
    """
    <style>
    .main .block-container {
        padding: 2rem 6rem;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Personalized Lipstick Recommendation Dashboard")

col1, spacer, col2 = st.columns([1.3, 0.2, 3.7])

with col1:
    st.header("Preferences")
    st.markdown(
        """
        ### Choose Your Method
        - Upload an image for automatic skin tone detection
        - Manually select your skin tone
        """
    )

    identification_method = st.radio(
        "How would you like to determine your skin tone?",
        [
            "Upload Image to identify Skin Tone",
            "Manual Selection by Color or Skin Tone",
        ],
    )

    detected_expert_tone = None
    detected_sephora_tone = None

    expert_mapping = create_id_name_mapping(expert_df, "Skin_ID", "Skin_Name")
    sephora_mapping = create_id_name_mapping(review_df, "Skin_ID_Sephora", "Skin_Tone_Name")

    if identification_method == "Upload Image to identify Skin Tone":
        detected_expert_tone, detected_sephora_tone = process_uploaded_image()
        tab_options = [
            "Expert Recommended Products",
            "Sephora user Recommended Products",
        ]
        tab = st.radio("Based on your Skin Tone choose:", tab_options)
    else:
        tab_options = [
            "Choose your favorite Color",
            "Choose your Skin Tone to get expert recommended product",
            "Choose your Skin Tone to get Sephora user recommended product",
        ]
        tab = st.radio("Choose your method:", tab_options)

    if tab in (
        "Expert Recommended Products",
        "Choose your Skin Tone to get expert recommended product",
    ):
        if identification_method == "Upload Image to identify Skin Tone" and detected_expert_tone:
            skin_tone_id = detected_expert_tone
            skin_tone_name = expert_mapping.get(skin_tone_id, f"Unknown (ID: {skin_tone_id})")
            st.info(f"Using detected skin tone: {skin_tone_name}")
        else:
            expert_options = format_selection_options(expert_mapping)
            selected_name = st.selectbox(
                "Select Your Skin Tone",
                list(sorted(expert_options.keys(), key=expert_options.get)),
                help="Choose your skin tone from the expert recommendation system",
                key="expert_select",
            )
            skin_tone_id = expert_options[selected_name]

    elif tab in (
        "Sephora user Recommended Products",
        "Choose your Skin Tone to get Sephora user recommended product",
    ):
        if identification_method == "Upload Image to identify Skin Tone" and detected_sephora_tone:
            skin_tone_id = detected_sephora_tone
            skin_tone_name = sephora_mapping.get(skin_tone_id, f"Unknown (ID: {skin_tone_id})")
            st.info(f"Using detected skin tone: {skin_tone_name}")
        else:
            sephora_options = format_selection_options(sephora_mapping)
            selected_name = st.selectbox(
                "Select Your Skin Tone",
                list(sorted(sephora_options.keys(), key=sephora_options.get)),
                help="Choose your skin tone from the Sephora system",
                key="sephora_select",
            )
            skin_tone_id = sephora_options[selected_name]
    else:  # Choose your favorite Color tab
        st.write("Select Color Group:")
        unique_colors = color_df.drop_duplicates(["ColorCluster", "ClusterColorDescription"])
        color_cluster_id = unique_colors.iloc[0]["ColorCluster"]

        color_container = st.container()
        cols = st.columns(3)

        if "selected_color_id" not in st.session_state:
            st.session_state.selected_color_id = color_cluster_id

        for idx, row in unique_colors.iterrows():
            col_idx = idx % 3
            color_hex = rgb_to_hex(row["cluster_R"], row["cluster_G"], row["cluster_B"])
            with cols[col_idx]:
                if st.button(
                    row["ClusterColorDescription"],
                    key=f"color_{row['ColorCluster']}",
                    help=f"Select {row['ClusterColorDescription']}",
                    type="secondary",
                ):
                    st.session_state.selected_color_id = row["ColorCluster"]
                st.markdown(
                    f"""
                    <div style="height: 10px; 
                         background-color: {color_hex}; 
                         margin-bottom: 10px; 
                         border: 1px solid #ddd;
                         border-radius: 2px;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if st.session_state.selected_color_id is not None:
            selected_row = unique_colors[unique_colors["ColorCluster"] == st.session_state.selected_color_id].iloc[0]
            color_hex = rgb_to_hex(selected_row["cluster_R"], selected_row["cluster_G"], selected_row["cluster_B"])
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; 
                     background-color: white; padding: 10px; 
                     border-radius: 4px; margin-top: 10px;">
                    <div style="width: 30px; height: 30px; 
                         background-color: {color_hex}; 
                         border: 1px solid #ddd; margin-right: 10px; 
                         border-radius: 3px;"></div>
                    <div>Selected Color: {selected_row['ClusterColorDescription']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            color_cluster_id = st.session_state.selected_color_id

    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=20,
        value=5,
        help="Adjust the number of product recommendations you'd like to see",
    )

with col2:
    if tab in (
        "Expert Recommended Products",
        "Choose your Skin Tone to get expert recommended product",
    ):
        st.subheader("Expert Recommended Lipsticks")
        expert_recs = get_expert_recommendations(skin_tone_id, n_recommendations)
        expert_recs = add_filters(expert_recs)
        formatted_df = format_dataframe(expert_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        add_visualizations(expert_recs)
        add_export_option(expert_recs)

    elif tab in (
        "Sephora user Recommended Products",
        "Choose your Skin Tone to get Sephora user recommended product",
    ):
        st.subheader("Top Rated by Users")
        review_recs = get_review_recommendations(skin_tone_id, n_recommendations)
        review_recs = add_filters(review_recs)
        formatted_df = format_dataframe(review_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        add_visualizations(review_recs)
        add_export_option(review_recs)

    else:
        st.subheader("Color Matched Recommendations")
        color_recs = get_color_recommendations(color_cluster_id, n_recommendations)
        color_recs = add_filters(color_recs)
        formatted_df = format_dataframe(color_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        add_visualizations(color_recs)
        add_export_option(color_recs)

st.markdown(
    """
    <style>
    table {
        width: 100% !important;
        margin-bottom: 1em;
        border-collapse: collapse;
    }
    th, td {
        text-align: left !important;
        padding: 12px !important;
        white-space: normal !important;
        border: 1px solid #ddd;
    }
    th {
        background-color: #f0f2f6 !important;
        font-weight: bold !important;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    td a {
        color: #1e88e5 !important;
        text-decoration: none !important;
    }
    td a:hover {
        text-decoration: underline !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
