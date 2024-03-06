import streamlit as st
import yaml
import os
from src.StockSeer.pipeline.prediction import PredictionPipeline
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path

# Set Streamlit page configuration
st.set_page_config(
    page_title="Stock Seer",
    page_icon="📈",
    layout="centered"
)

st.header("Stock Seer",divider='rainbow')
st.subheader("your AI powered stock price prediction app")

with st.form(key="prediction_form",clear_on_submit=True):
    
    ticker = st.text_input("Enter stock ticker:",placeholder="For example, AAPL")
    n_sessions = st.number_input("Enter the number of future sessions to predict:", min_value=1, value=1)

    submitted = st.form_submit_button("Submit")

if submitted:
  
    with open(Path('config\config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    config['data_ingestion']['ticker'] = ticker

    # Write the updated YAML file
    with open(Path('config\config.yaml'), 'w') as file:
        yaml.dump(config, file)
    
    os.system("python main.py")

    obj = PredictionPipeline()
    predicted_value = obj.Predict(n_sessions)
    predicted_value = np.round(predicted_value[:,0],decimals=2)

    sessions = np.arange(1, len(predicted_value)+1, 1, dtype=int)

    st.subheader("Chart of Session VS predicted Closing Price",divider='rainbow')

    fig = px.area(x=sessions, y=predicted_value,
              labels={'x': 'Session Number', 'y': 'Stock Price'},
              line_shape='linear',  
              color_discrete_sequence=['#636EFA'],  
              )
    st.plotly_chart(fig,theme ='streamlit')

    st.header("Session wise predicted closing price",divider="violet")

    df = pd.DataFrame({"Session Number":sessions,
                  "Predicted Closing price":predicted_value})
    df["Predicted Closing price"] = df["Predicted Closing price"].round(2)
    st.dataframe(df,hide_index =True)

    