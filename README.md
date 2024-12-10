# Lipshade Lab - Lipstick Recommendation System

Lipshade Lab is an interactive web application that helps users find the perfect lipstick shades based on their skin tone. The system takes user input either through automatic skin tone detection (via Google Mediapipe by uploaded image) or manual skin tone selection. The application recommends lipstick shades based on expert suggestions, user reviews, and color clustering.

This repository contains the code to run the lipstick recommendation system using Streamlit. It also provides the necessary scripts to handle data collection, preprocessing, and recommendation logic.

## Getting Started

### Installation
If you have not yet installed the necessary dependencies for the project, start by installing them via the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Navigate into the project directory
```bash
git clone https://github.com/NoraStella-Chen/lipshade-lab.git

cd 3.\ Lipstick\ Recommendation 
```
### Running the Application
From the project directory, start the application by running:
```bash
streamlit run app.py
```
### Accessing the Application

Once the server is running, open http://localhost:8501 in your web browser to access the lipstick recommendation interface.

We also host it in Azure: [Link to Web App](https://lipstick-app.azurewebsites.net/)

## Data Access Statement

### Overview
This repository does not include the raw or processed datasets due to their large size. Please refer to the links below for data access. All data is owned by Sephora and is presented here strictly for academic, non-commercial use. Users must comply with Sephora’s terms of use and adhere to all applicable licenses.

### Data Access
- **Raw Data:** [Link to Raw Data](https://drive.google.com/drive/folders/1YpIC5FMkOywOepyriTlxhTys8FUpzqhg?usp=drive_link)
- **Preprocessed Data:** [Link to Preprocessed Data](https://drive.google.com/drive/folders/1qhZ-I2TCP2QRb5FCXTb6z-sKE_PkkWnG?usp=drive_link)

### Data Collection
- **Source:** Data was collected from [Sephora’s website](https://www.sephora.com) in October 2024.
- **Compliance:** All data collection activities followed Sephora’s published terms of use and the guidelines specified in the site’s `robots.txt` file.

### Data Ownership and Licensing
- The original data is the property of Sephora.
- The data is provided exclusively for non-commercial, academic use.
- No redistribution of the raw datasets is planned. Any findings or analyses derived from them will be shared only in aggregate form.

### Scraping Methods
#### Product Pages
- Approximately 300 publicly available product pages were collected via standard HTTP requests.
- Only general, publicly visible product information was retrieved.

#### API for Reviews
- Ratings and review data were obtained from Sephora’s publicly accessible API.
- No login credentials or user-specific information were collected.
- The data collection process followed best practices to minimize server impact.

## Recommendation Datasets
### Overview
To build a recommendation system that takes into account skin tone, personal preferences, and product qualities, we created the following datasets:

- Expert Recommendation Dataset: Contains lipstick product suggestions for each skin tone, based on expert recommendations.
- Sephora Review-Based Recommendation Dataset: Contains insights from users on which lipstick products work well for different skin tones.
- Color Clustering Recommendation Dataset: Uses color clustering to group similar lipstick shades and allows users to explore products based on color similarity.

### Data Pipeline
- Product Collection: We start by extracting a list of lipstick products from Sephora's lipstick catalog page using a browser. We then extract detailed information from each product page, including likes, total reviews, and prices, using the product page URL.
- Image Extraction: We download images using URLs obtained from the product pages. These images include product covers and SKU swatch images.
- Review Extraction: We gather reviews using the Bazaarvoice API and web scrape 14 JSON files to consolidate review information into a single dataset.

### Data Transformation
After collecting the data, we merge it into a single dataset containing detailed product information, user reviews, and images. We filter out records with missing key information (e.g., skin tone, missing images) and reclassify the skin tone field for consistency.

In the end, we have:
215 lipstick products
1108 SKU data entries
53,539 reviews

### Exploratory Data Analysis & Recommendation Datasets Preparation
We created a word cloud to identify key review terms. Customers tend to focus on color shades, texture (glossy vs matte), skin tone compatibility, long-lasting properties, and packaging, as shown in the Appendix.

### Recommendation Dataset Enrichment
We enriched the three datasets with sentiment analysis of reviews using the Llama3.2 model. The sentiment analysis was conducted by summarizing each product review into short phrases, starting with an emoji.

## Recommendation System
In the recommendation system, users can:

- Upload an image for automatic skin tone detection, or
- Manually select their skin tone or preferred color for recommendations.

### Automatic Skin Tone Detection
Once an image is uploaded, our skin tone detection model generates two outputs: skin_tone_id and skin_id_sephora. Based on these, users can receive expert or Sephora user recommendations.

### Manual Skin Tone or Color Selection
Users can also manually select their skin tone for expert recommendations or choose a favorite color for personalized product suggestions.

## Acknowledgments
This project would not be possible without the help of the Sephora dataset and review data, as well as the useful resources from beauty experts and KOLs that guided the creation of personalized lipstick shade recommendations.

## License
This project is for academic, non-commercial use only. All data and information from Sephora is used in compliance with their terms of service.
