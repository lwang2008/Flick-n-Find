import streamlit as st
import pandas as pd
if st.button("Show Image Database"):
    st.write("### Image Database")
    imageDB = pd.read_csv("imageDB.csv")
    st.write(imageDB)

if st.button("Clear Database"):
    st.write("### Clearing Database")
    imageDB = pd.read_csv("imageDB.csv")
    imageDB = imageDB[0:0]
    imageDB.to_csv("imageDB.csv", index=False)