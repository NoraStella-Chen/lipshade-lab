import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Load the data files
file_path_expert_skintone = "expertRecommendation_top_10_skus_per_RBG.xlsx"
file_path_review_skintone = "skinToneRecommendation_top_20_skus_per_skinTone.xlsx"
file_path_colorcluster = "colorClusterRecommendation_top_20_skus_per_colorCluster.xlsx"

expert_df = pd.read_excel(file_path_expert_skintone)
review_df = pd.read_excel(file_path_review_skintone)
color_df = pd.read_excel(file_path_colorcluster)

def get_recommendations(df, filter_column, filter_value, n_recommendations=5, source_name="Recommendation"):
    recommendations = df[df[filter_column] == filter_value]
    recommendations = recommendations.sort_values('skuTotalReviews', ascending=False).head(n_recommendations)
    recommendations['Source'] = source_name
    return recommendations

def format_dataframe(df):
    # Create a copy of the dataframe with selected columns
    formatted_df = df[[
        'productID', 'skuID', 'brandName', 'displayName', 
        'currentSku_listPrice', 'color_description', 
        'skuRating', 'skuTotalReviews', 'URL'
    ]].copy()
    
    # Format price - handle existing dollar signs
    def format_price(x):
        try:
            if pd.notnull(x):
                price_value = str(x).replace('$', '').replace(',', '').strip()
                return f"${float(price_value):.2f}"
            return ""
        except (ValueError, TypeError):
            return str(x)
    
    # Format rating
    def format_rating(x):
        try:
            if pd.notnull(x):
                return f"{float(x):.1f}/5"
            return ""
        except (ValueError, TypeError):
            return str(x)
    
    # Apply formatting
    formatted_df['currentSku_listPrice'] = formatted_df['currentSku_listPrice'].apply(format_price)
    formatted_df['skuRating'] = formatted_df['skuRating'].apply(format_rating)
    formatted_df['skuID'] = formatted_df['skuID'].astype(str).str.replace(r'\D', '', regex=True)
    
    # Make URLs clickable
    formatted_df['URL'] = formatted_df['URL'].apply(
        lambda x: f'<a href="{x}" target="_blank">View Product</a>' if pd.notnull(x) else ""
    )
    
    # Rename columns for display
    column_rename = {
        'productID': 'Product ID',
        'skuID': 'SKU',
        'brandName': 'Brand',
        'displayName': 'Product Name',
        'currentSku_listPrice': 'Price',
        'color_description': 'Color',
        'skuRating': 'Rating',
        'skuTotalReviews': 'Reviews',
        'URL': 'Link'
    }
    formatted_df.rename(columns=column_rename, inplace=True)
    
    return formatted_df

def get_expert_recommendations(skin_tone_id, n_recommendations=5):
    return get_recommendations(expert_df, 'Skin_ID', skin_tone_id, n_recommendations, "Expert Choice")

def get_review_recommendations(skin_id_sephora, n_recommendations=5):
    return get_recommendations(review_df, 'Skin_ID_Sephora', skin_id_sephora, n_recommendations, "Top Rated by Users")

def get_color_recommendations(color_cluster_id, n_recommendations=5):
    return get_recommendations(color_df, 'ColorCluster', color_cluster_id, n_recommendations, "Color Match")

# Streamlit app layout
st.title("Personalized Lipstick Recommendation Dashboard")

# Create two columns for layout
col1, col2 = st.columns([1, 4])

# Sidebar contents
with col1:
    st.header("Preferences")
    tab_options = ["Expert Recommendations", "User Reviews", "Color Matching"]
    tab = st.radio("Recommendation Type", tab_options)
    
    # Show different skin tone options based on selected tab
    if tab == "Expert Recommendations":
        skin_tone_id = st.selectbox("Select Expert Skin Tone ID", sorted(expert_df['Skin_ID'].unique()))
    elif tab == "User Reviews":
        # Sort Skin_ID_Sephora values numerically
        sorted_sephora_ids = sorted(review_df['Skin_ID_Sephora'].unique(), key=lambda x: float(x) if pd.notnull(x) else float('-inf'))
        skin_tone_id = st.selectbox("Select Sephora Skin Tone ID", sorted_sephora_ids)
    else:
        # Sort Color Cluster values numerically
        sorted_color_clusters = sorted(color_df['ColorCluster'].unique(), key=lambda x: float(x) if pd.notnull(x) else float('-inf'))
        color_cluster_id = st.selectbox("Select Color Cluster", sorted_color_clusters)
    
    n_recommendations = st.slider("Number of Recommendations", 1, 20, 5)

# Main content
with col2:
    if tab == "Expert Recommendations":
        st.subheader("Expert Recommended Lipsticks")
        expert_recs = get_expert_recommendations(skin_tone_id, n_recommendations)
        formatted_df = format_dataframe(expert_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
    elif tab == "User Reviews":
        st.subheader("Top Rated by Users")
        review_recs = get_review_recommendations(skin_tone_id, n_recommendations)
        formatted_df = format_dataframe(review_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
    else:
        st.subheader("Color Matched Recommendations")
        color_recs = get_color_recommendations(color_cluster_id, n_recommendations)
        formatted_df = format_dataframe(color_recs)
        st.write(formatted_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Add custom CSS for better table styling
st.markdown("""
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
    """, unsafe_allow_html=True)