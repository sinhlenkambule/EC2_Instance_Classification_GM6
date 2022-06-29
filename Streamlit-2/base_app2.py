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

# Data dependencies
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu 
from PIL import Image

# This command allows the app to use wide mode of the screen.
st. set_page_config(layout="wide")

# Use local CSS to sort the styling of the contact form
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app

def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Function to access the json files of the lottie animations
	def load_lottieurl(url):
				r = requests.get(url)
				if r.status_code != 200:
					return None
				return r.json()

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	with st.sidebar:
		selected = option_menu(
			menu_title = "Main Menu",
			menu_icon="list", 
			options = ['Prediction', 'Information', 'Visualisations', 'Get in touch with us!'],
			icons = [ "search","info-circle", "bar-chart-line", "envelope"],
			)
		
	logo = Image.open("images/logo_2.png")
	st.sidebar.image("images/logo_2.png", use_column_width=True)
	
# Building out the predication page
	if selected == "Prediction":
		left_column, right_column = st.columns(2)
		with left_column:
			st.info("Prediction with ML Models")

			# Creating a text box for user input		
			tweet_text = st.text_area("Enter your tweet below:", "Type Here")
		
			# Dropdown menu for different models
			models = ["Logistic Regression", "KNN", "Linear SVC"]
			model_choice = st.selectbox("Select classifier.", models)

			# Implemeting the selected model
			if model_choice == "Logistic Regression":
				predictor = joblib.load(open(os.path.join("resources/Logistic.pkl"),"rb"))

			if model_choice == "KNN":
				predictor = joblib.load(open(os.path.join("resources/KNN.pkl"),"rb"))

			if model_choice =="Linear SVC":
				predictor = joblib.load(open(os.path.join("resources/SVC.pkl"),"rb"))


			if st.button("Classify"):

			# Getting the predictions
				def get_keys(val, my_dict):
					for key, value in my_dict.items():
						if val == value:
							return key
			
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				prediction = predictor.predict(list(vect_text))
				prediction_labels = {'Anthropogenic':-1,'Neutral':0,'Prominent':1,'News':2}

				final_result = get_keys(prediction, prediction_labels)
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				# predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				# prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Tweet Categorized as: {}".format(final_result))

		with right_column:
			predict_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_l5qvxwtf.json")
			st_lottie(predict_animation, height=300, key="coding2")

	# Building out the "Information" page
	if selected == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("The goal of this project is to develop a machine learning model that can rank whether or not someone believes in climate change based on new Twitter data. By providing an accurate and robust solution to this task, we are basically helping companies gain access to a large pool of customer views across many demographic and geographic groups, enabling them to gain new insights and better inform future marketing initiatives.")
	
		climate_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_yw7dq20d.json")
		st_lottie(climate_animation, height=300, key="coding")
		
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page



	# Building the "Visualisations" page
	if selected == "Visualisations":
		st.info("##### For this section we will explore the distribution of the data using different visualisation plots")
		st.write("---")

		bar_graph = Image.open('images/bar_graph.png')
		st.write("##### Bar graph showing tweets per sentiment")
		st.image(bar_graph, width = None)
		st.write("""
					#### Quick overview of the data above:
					* There is a strong imbalance amongst the sentiments of the tweets.
					* Based on the information/sentiments displayed by our histogram, we can see that the vast majority lies within the "1"-sentiment (the tweet that supports the belief of man-made climate change).
					* On a few distribution is shown under the "-1"-sentiment (the tweet that does not supports the belief of man-made climate change)
				
				""")
		st.write("---")
		st.write("#")

		pie_chart = Image.open('images/pie_chart2.png')
		st.write("##### Pie chart showing the percentages of tweets per sentiment")
		st.image(pie_chart)
		st.write("""
					#### Quick overview of the data above:
					* The distribution is very clear on the pie chart showing the percentages of tweets for every sentiment,
					* A vast majority lies within the Prominen sentiment, dominating the chart with 54% contribution.
					* While Anthropogenic sentiment only contrinutes with 8%.
					* Through the precentages displayed in the pie-chart, we can observe that the majority of people believes in man-made climate change. 
					
				""")

				
	# Building the "Get in touch with us" page
	if selected == "Get in touch with us!":
		left_column, right_column = st.columns(2)
		with left_column:
			st.info("##### About Us")
			st.write("""We are Explore Tech SA. 
			\nAn organization, based in Southern Africa, with a passion for solving problems by using the unique skill set of our team. Explore Tech SA was founded in 2016, is among the largest IT & Business consulting services firm in Africa, We are insights-driven and outcome-based to help accelerate returns on your IT and business investments. In all we do, our goal is to build trusted relationships through client proximity, providing industry and technology expertise to help you meet the needs of your customers and citizens.""")

		with right_column:
			# Loading the animation in the "Get in touch with us!" section.
			contact_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_7wwm6az7.json")
			st_lottie(contact_animation, height=300, key="coding")

		st.write("---")

		# Details of the team
		with st.container():
			st.info('##### Connect with us:')

			left_column, right_column = st.columns(2)
			with left_column:
				st.caption("### Ditheto Mathekga")
				st.write("""Chief Executive Officer
					\nditheto.mathekga@exploretech.co.za""")
					 
				st.caption("### Sinhle Nkambule")
				st.write("""Web App Developer
					\nsinhle.nkambule@exploretech.co.za""")

				st.caption("### Donovan Makate")
				st.write("""Machine Learning Engineer
					\nmokwale.makate@exploretech.co.za""")
			
			with right_column:
				st.caption("### Njabulo Preysgod Nsibande")
				st.write("""Data Science Manager
					\nnjabulo.nsibande@exploretech.co.za""")

				st.caption("### Nthapeleng Linah Raphela")
				st.write("""Senior Data Analyst
					\nlinah.raphela@exploretech.co.za""")					  

				st.caption("### Bunga Never Baloyi")
				st.write("""Data Scientist
					\nbunganever.valoyi@exploretech.co.za""")
				

		st.write("---")

		# Contact form for queries.
		with st.container():
			st.info("##### Send your enquiry direct to our company admin team")

			# Documention: https://formsubmit.co/
			contact_form =( """
			<form action="https://formsubmit.co/sinhlenkambule78@gmail.com" method="POST">
				<input type="hidden" name="_captcha" value="false">
				<input type="text" name="name" placeholder="Your name" required>
				<input type="email" name="email" placeholder="Your email" required>
				<textarea name="message" placeholder="Your message here" required></textarea>
				<button type="submit">Send</button>
			</form>
			""")

			left_column, right_column = st.columns(2)
			with left_column:
				st.markdown(contact_form, unsafe_allow_html=True)
			with right_column:
				st.empty()
		


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
