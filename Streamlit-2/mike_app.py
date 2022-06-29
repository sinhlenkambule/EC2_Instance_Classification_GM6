"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu

# Data dependencies
import pandas as pd

# Vectorizer
# news_vectorizer = open("resources/vectorizer.pkl","rb")
# tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	
	with st.sidebar:
		selection=option_menu(
			menu_title="Main Menu",
			options=["Prediction", "Information","Contact Us","About Us"],
			icons=["bar-chart-line","info-circle","envelope","people-fill"],
			menu_icon="cast",
			default_index=0,
		)
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		select_model=option_menu(
			menu_title="Select Classifier",
			options=["Decision Tree", "KNeighbors","Logistic_Regression","SVC"],
			icons=["arrow-right-circle","arrow-right-circle","arrow-right-circle","arrow-right-circle"],
			menu_icon="hand-index-thumb-fill",
			default_index=0,		
		)

		
		if select_model == "Decision Tree":
			predictor = joblib.load(open(os.path.join("resources/DecisionTreeClassifier.pkl"),"rb"))

		if select_model == "KNeighbors":
			predictor = joblib.load(open(os.path.join("resources/KNeighborsClassifier.pkl"),"rb"))

		if select_model =="Logistic_Regression":
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))

		if select_model == "SVC":
			predictor = joblib.load(open(os.path.join("resources/SVC.pkl"),"rb"))

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","")
		
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()	
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

			
	if selection == "Contact Us":
		st.header(":mailbox: Get In Touch With Us!")
		contact_form = """
				<form action="https://formsubmit.co/mikelacoste25@gmail.com" method="POST">
					<input type="hidden" name="_captcha" value="false">
					<input type="text" name="name" placeholder="Your name" required>
					<input type="email" name="email" placeholder="Your email" required>
					<textarea name="message" placeholder="Your message here"></textarea>
					<button type="submit">Send</button>
				</form>
				"""

		st.markdown(contact_form, unsafe_allow_html=True)

				# Use Local CSS File
		def local_css(file_name):
			with open(file_name) as f:
				st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

		local_css("style/style.css")
	if selection == "About Us":
		st.header("Who we are")


		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
