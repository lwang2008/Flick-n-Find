import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Search for Lost Items",
    page_icon="🔍",
    layout="wide"
)
st.title("Search for Lost Items")

import preprocess
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# Load the Universal Sentence Encoder directly from TF Hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def embed(input): # takes in a string input
    return model([input]).numpy().reshape(1, -1)

# Add custom CSS to disable image popup
st.markdown("""
    <style>
    .stImage img {
        pointer-events: none;
    }
    .stImage div {
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

def cos_sim(v1, v2):
    return cosine_similarity(v1, v2)[0][0]

query = ""

def search():
    global query
    try:
        imageDB = pd.read_csv("imageDB.csv")
        if query == "":
            # show everything in the database
            #lost items
            db = imageDB.iloc[::-1].reset_index(drop=True)
            db1 = db[db["status"] == "lost"]
            db1 = db1.reset_index(drop=True)
            display_results(db1)

        else:
            query = preprocess.preprocess(query)
            actual_vector = embed(query)
            imageDB["USE_vector"] = imageDB["keywords"].apply(embed)
            imageDB["cos_sim"] = imageDB["USE_vector"].apply(lambda x: cos_sim(actual_vector, x))
            db = imageDB.sort_values(by='cos_sim', ascending=False)
            db1 = db[db["status"] == "lost"]
            db1 = db1.reset_index(drop=True)
            display_results(db1)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def foundItem(row):
    global query
    ind = row["ind"] # finding the unique ind
    imageDB = pd.read_csv("imageDB.csv") # reading the imageDB
    imageDB.loc[imageDB['ind'] == ind, 'status'] = 'found'  # set the imageDB row status
    imageDB.to_csv("imageDB.csv", index=False)
    if f"button_{ind}" in st.session_state:
        del st.session_state[f"button_{ind}"]
    st.experimental_rerun()

def display_results(db):
    num_columns = 3  # Number of cards per row
    columns = st.columns(num_columns)
    
    for i, row in db.iterrows():
        col = columns[i % num_columns]
        with col:
            try:
                image = Image.open(row['filepath'])
                st.image(image, width=250)
            except FileNotFoundError:
                st.write("Image not found.")
            
            st.write(f"**Location:** {row['location']}")
            st.write(f"**Time Found:** {row['time']}")
            st.write(f"**Date Found:** {row['date']}")
            contact_info = "N/A" if np.isnan(row['contact']) else str(int(row['contact']))
            st.write(f"**Contact Info:** {contact_info}")
            
            if st.button("Found item", key=f"button_{row['ind']}"):
                foundItem(db.loc[i])

            st.markdown("</div>", unsafe_allow_html=True)

query = st.text_input("Enter your search query:")

# searching
results = search()
