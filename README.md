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


