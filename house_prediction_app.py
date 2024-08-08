import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
import joblib

# Load the dataset
try:
    house_price_dataset = fetch_california_housing()
except PermissionError as e:
    st.error("Permission error while accessing the dataset file. Please ensure no other application is using the file and try again.")
    st.stop()

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe['price'] = house_price_dataset.target

# Split the data
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Check if model is already saved, otherwise train and save it
model_path = "house_price_model.joblib"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = XGBRegressor(tree_method='auto')
    model.fit(X_train, Y_train)
    joblib.dump(model, model_path)

# Mapping of city names to their approximate latitude and longitude (Indian cities)
city_to_lat_lon = {
    "Mumbai": (19.076, 72.877),
    "Delhi": (28.704, 77.102),
    "Bangalore": (12.971, 77.594),
    "Kolkata": (22.572, 88.363),
    "Chennai": (13.082, 80.27),
    "Hyderabad": (17.385, 78.486),
    "Pune": (18.520, 73.856),
    "Ahmedabad": (23.022, 72.571),
    "Surat": (21.170, 72.831),
    "Jaipur": (26.912, 75.787),
    "Lucknow": (26.846, 80.946),
    "Kanpur": (26.449, 80.331),
    "Nagpur": (21.146, 79.088),
    "Visakhapatnam": (17.686, 83.218),
    "Indore": (22.719, 75.857),
    "Thane": (19.218, 72.978),
    "Bhopal": (23.259, 77.412),
    "Patna": (25.594, 85.137),
    "Vadodara": (22.307, 73.181),
    "Ghaziabad": (28.669, 77.453),
    "Ludhiana": (30.901, 75.857),
    "Agra": (27.176, 78.008),
    "Nashik": (19.997, 73.789),
    "Faridabad": (28.408, 77.317),
    "Meerut": (28.984, 77.706),
    "Rajkot": (22.303, 70.802),
    "Kalyan-Dombivli": (19.238, 73.127),
    "Vasai-Virar": (19.425, 72.822),
    "Varanasi": (25.317, 82.973),
    "Srinagar": (34.083, 74.797),
    "Aurangabad": (19.876, 75.343),
    "Dhanbad": (23.795, 86.429),
    "Amritsar": (31.634, 74.872),
    "Navi Mumbai": (19.033, 73.029),
    "Allahabad": (25.435, 81.846),
    "Ranchi": (23.344, 85.309),
    "Haora": (22.576, 88.318),
    "Gwalior": (26.218, 78.182),
    "Jabalpur": (23.181, 79.986),
    "Coimbatore": (11.017, 76.957),
    "Vijayawada": (16.506, 80.648),
    "Madurai": (9.925, 78.119),
    "Guwahati": (26.144, 91.736),
    "Chandigarh": (30.733, 76.779),
    "Hubli-Dharwad": (15.364, 75.124),
    "Mysore": (12.295, 76.639),
    "Raipur": (21.251, 81.629)
    # Add other cities here
}

# City-specific scaling factors to adjust the price prediction
city_scaling_factors = {
    "Mumbai": 1.5,
    "Delhi": 1.4,
    "Bangalore": 1.3,
    "Kolkata": 1.2,
    "Chennai": 1.2,
    "Hyderabad": 1.3,
    "Pune": 1.3,
    "Ahmedabad": 1.1,
    "Surat": 1.1,
    "Jaipur": 1.0,
    "Lucknow": 1.0,
    "Kanpur": 0.9,
    "Nagpur": 1.0,
    "Visakhapatnam": 1.1,
    "Indore": 1.0,
    "Thane": 1.2,
    "Bhopal": 0.9,
    "Patna": 0.9,
    "Vadodara": 0.9,
    "Ghaziabad": 1.0,
    "Ludhiana": 0.9,
    "Agra": 0.8,
    "Nashik": 0.9,
    "Faridabad": 1.0,
    "Meerut": 0.9,
    "Rajkot": 0.8,
    "Kalyan-Dombivli": 1.1,
    "Vasai-Virar": 1.1,
    "Varanasi": 0.8,
    "Srinagar": 0.9,
    "Aurangabad": 0.9,
    "Dhanbad": 0.7,
    "Amritsar": 0.8,
    "Navi Mumbai": 1.4,
    "Allahabad": 0.8,
    "Ranchi": 0.7,
    "Haora": 1.1,
    "Gwalior": 0.7,
    "Jabalpur": 0.7,
    "Coimbatore": 1.0,
    "Vijayawada": 1.0,
    "Madurai": 0.9,
    "Guwahati": 0.9,
    "Chandigarh": 1.0,
    "Hubli-Dharwad": 0.8,
    "Mysore": 0.9,
    "Raipur": 0.8
    # Add other cities here
}

# Streamlit App
st.title("House Price Prediction App (India)")

# Input features form with user-friendly inputs
st.subheader("Input Features")
with st.form(key='prediction_form'):
    city = st.selectbox("City", list(city_to_lat_lon.keys()))
    avg_income = st.number_input("Average Income (in Rs. Lakhs)", min_value=0.0, max_value=150.0, value=35.0, step=1.0)
    house_age = st.number_input("House Age (in years)", min_value=0, max_value=100, value=20, step=1)
    avg_rooms = st.number_input("Average Number of Rooms", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
    avg_bedrooms = st.number_input("Average Number of Bedrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    population = st.number_input("Population", min_value=0, max_value=400000, value=10000, step=100)
    avg_occupancy = st.number_input("Average Occupancy", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    latitude, longitude = city_to_lat_lon[city]
    input_data = np.array([[avg_income, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude]])
    prediction = model.predict(input_data)[0] * 100000
    city_scaling_factor = city_scaling_factors[city]
    adjusted_prediction = prediction * city_scaling_factor

    st.write(f"Predicted House Price in {city}: Rs. {adjusted_prediction:,.2f}")

    # Generate synthetic past 2 years of data for house prices with fluctuations
    num_months = 24
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=num_months, freq='M')
    price_changes = np.random.normal(0.01, 0.03, num_months)  # Mean = 1%, std = 3% for monthly change
    synthetic_prices = adjusted_prediction * np.cumprod(1 + price_changes)

    # Create a DataFrame for past house prices
    past_prices_df = pd.DataFrame({
        'Date': date_range,
        'Price': synthetic_prices
    })

    # Plot the past house prices using Altair
    line_chart = alt.Chart(past_prices_df).mark_line(point=True).encode(
        x='Date:T',
        y=alt.Y('Price:Q', title='Price (in Rs.)'),
        tooltip=['Date:T', 'Price:Q']
    ).properties(
        title=f'House Price Trend in {city} Over the Past 2 Years',
        width=700,
        height=400
    ).configure_title(
        fontSize=20,
        anchor='start'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_mark(
        color='steelblue'
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)
