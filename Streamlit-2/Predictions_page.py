# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu

# Building out the prediction page
tweet_text = st.text_area('Enter Text (max. 150 characters):') 
all_ml_models = ["Logistic Regression","KNN"," Linear SVC"]
model_choice = st.selectbox("Select a ML model to classify your Tweet.", all_ml_models)