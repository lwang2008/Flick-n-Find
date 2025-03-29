import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import uuid


# Streamlit multi-page setup
st.set_page_config(page_title="Community Lost & Found", layout="wide")
st.title("Community Lost & Found")




# Introduction section
st.markdown(
    """
    Welcome to **Community Lost & Found**, a platform dedicated to reconnecting lost items with their rightful owners. 
    Services such as Nextdoor do not have a dedicated lost and found section, and any such posts will be lost in the stream of daily posts. 
    Our goal is to make it easier for people to report and search for lost belongings in their community.

    ### How it works:
    1. **Found an item?** Take a picture and upload it here. Our system will scan the image and generate relevant keywords to describe it. Include details such as date, time, and location. Add contact details.
    2. **Lost something?** Use our search feature to look for your item. Our AI will match your search to the items in our database. Contact the person who found your item. 

    Together, let's create a helpful and supportive community for lost and found items!
    """
)

# Call-to-action buttons
st.markdown("### Get Started")

col1, col2 = st.columns(2)
with col1:
    result=st.button("I Found Something")
    if result:
        st.switch_page("pages/upload.py")  # Navigate to the upload page
with col2:
    result=st.button("I Lost Something")
    if result:
        st.switch_page("pages/search.py")  # Navigate to the search page

# Footer
st.markdown("""
    ---
    Created by Helpful Hackers!
""")

