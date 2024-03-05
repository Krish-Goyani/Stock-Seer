import streamlit as st
import yaml
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Stock Seer",
    page_icon="📈",
    layout="centered"
)

st.title("Stock Seer - your AI powered stock price prediction app")

with st.form(key="prediction_form",clear_on_submit=True):
    
    ticker = st.text_input("Enter stock ticker:", "AAPL")
    n_sessions = st.number_input("Enter the number of future sessions to predict:", min_value=1, value=1)

    submitted = st.form_submit_button("Submit")

if submitted:
  
    with open('config\config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['data_ingestion']['ticker'] = ticker

    # Write the updated YAML file
    with open('config\config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    os.system("python main.py")
    