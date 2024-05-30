import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import joblib



warnings.filterwarnings("ignore")

st.title('Subscription Prediction Web App')

import logging
logger = logging.getLogger(__name__)



# Input fields
age = st.number_input('Enter your age', step=1)
job = st.selectbox('select your job', ('blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'))
marital = st.selectbox('Select your marital status', ('married', 'single', 'divorced'))
education = st.selectbox('Select your education', ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'))
default = st.selectbox('Select if you have credit in default or not', ('yes', 'no', 'unknown'))
housing = st.selectbox('Select if you have a housing loan or not', ('yes', 'no', 'unknown'))
loan = st.selectbox('Select if you have a personal loan', ('yes', 'no', 'unknown'))
contact = st.selectbox('Select your contact communication type', ('cellular', 'telephone'))
month = st.selectbox('Select when was your last contact', ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
day_of_week = st.selectbox('Select your last contact day of the week', ('mon', 'tue', 'wed', 'thu', 'fri'))
duration = st.number_input('enter your last contact duration in seconds', step=1)
campaign = st.number_input('Enter the number of contacts performed', step=1)
pdays = st.number_input('Enter number of days passed after last contacted from a previous campaign', step=1)
if pdays == 0:
    pdays = 999
previous = st.number_input('Enter the number of contacts performed before this campaign', step=1)
poutcome = st.selectbox('Eelect outcome of the previous marketing campaign', ('success', 'failure', 'nonexistent'))
emp_var_rate = st.number_input('Enter employment variation rate')
cons_price_idx = st.number_input('Enter consumer price index')
cons_conf_idx = st.number_input('Enter consumer confidence index')
euribor3m = st.number_input('Enter euribor 3 month rate')
nr_employed = st.number_input('Enter number of employees')

# Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribor3m],
    'nr.employed': [nr_employed]
})

def scale(a):

        scaler = joblib.load('scaler.joblib')

        numerical_col = a.select_dtypes(include=["int64","float64"]).columns
        a[numerical_col] = scaler.transform(a[numerical_col])

        return a


# Load the model and preprocessing pipeline
with open('trained_model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

def predict_display(a):
    x = scale(a)
    # Apply preprocessing and prediction
    input_predictions = model_pipeline.predict(x)

    # Display the prediction
    if input_predictions == 0:  # Assuming 'no' is encoded as 0
        st.error(f'No')
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.write(' ')
        with col2:
            st.write(' ')
        with col3:
            st.write(' ')
        with col4:
            st.write(' ')
        with col5:
            st.write(' ')
        with col6:
            st.write(' ')
        with col7:
            st.write(' ')
    else:
        st.success(f'yes')
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.write(' ')
        with col2:
            st.write(' ')
        with col3:
            st.write(' ')




if st.button('Press me'):
    data = pd.read_csv("C:/Users\plaoz\Desktop/veri bilim\datasets/bank-additional.csv", delimiter= ";", quotechar= '"')
    input_data["y"] = False

    data = pd.concat([input_data,data], ignore_index= True)
    df = pd.get_dummies(data, drop_first=True)
    df = df[df.index == 0]
    predict_display(df)