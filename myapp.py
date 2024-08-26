# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:35:03 2024

@author: PHD4
"""

import streamlit as st

# Title for your app
st.title("My First Streamlit App")

# Text input
name = st.text_input("Enter your name:")

# Display greeting when the user enters a name
if name:
    st.write(f"Hello, {name}!")
