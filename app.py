import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Airline Fare Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom Styling (White Theme)
# -------------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    html, body, [class*="css"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    .main {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 0 15px rgba(0,0,0,0.05);
        width: 90%;
        max-width: 850px;
        margin: auto;
    }
    h1, h3 {
        color: #003366;
        text-align: center;
        font-weight: 700;
    }
    .stButton>button {
        display: block;
        margin: 35px auto 0 auto !important;
        background-color: #0066cc !important;
        color: white !important;
        border-radius: 8px !important;
        height: 50px !important;
        width: 240px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #004c99 !important;
        transform: scale(1.03);
    }
    .footer {
        text-align: center;
        color: #555;
        margin-top: 40px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Title
# -------------------------------
st.title("Airline Fare Prediction")
st.markdown("### Predict your flight fare using Machine Learning")

# -------------------------------
# Load Data and Train Model
# -------------------------------
@st.cache_data
def load_data_and_train_model():
    df = pd.read_csv("airlines_flights_data.csv")
    df.drop(columns=['index', 'flight'], inplace=True)

    cat_cols = ['airline', 'source_city', 'departure_time', 'stops',
                'arrival_time', 'destination_city', 'class']

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('price', axis=1)
    y = df['price']

    # Train a lightweight Random Forest model
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return le_dict, rf

try:
    le_dict, rf = load_data_and_train_model()
except Exception as e:
    st.error("⚠️ Error loading data or training model. Please ensure CSV file is in the same folder.")
    st.stop()

# -------------------------------
# User Input Section
# -------------------------------
st.markdown("## Enter Flight Details")

col1, col2 = st.columns(2)

clean_airlines = [a.replace("_", " ") for a in le_dict['airline'].classes_]

with col1:
    airline = st.selectbox("Airline", ["Select"] + clean_airlines)
    source = st.selectbox("Departure City", ["Select"] + list(le_dict['source_city'].classes_))
    destination = st.selectbox("Arrival City", ["Select"] + list(le_dict['destination_city'].classes_))
    travel_class = st.selectbox("Class", ["Select"] + list(le_dict['class'].classes_))

with col2:
    stops = st.radio("Number of Stops", ["No Stop", "1 Stop", "2 or More Stops"], index=1)
    departure = st.selectbox("Departure Time", ["Select"] + [s.replace("_", " ") for s in le_dict['departure_time'].classes_])
    arrival = st.selectbox("Arrival Time", ["Select"] + [s.replace("_", " ") for s in le_dict['arrival_time'].classes_])
    duration = st.slider("Duration (in hours)", 0.5, 30.0, 5.0, 0.5)
    days_left = st.number_input("Days Left before Departure", 1, 60, 15)

# -------------------------------
# Validation Function
# -------------------------------
def validate_inputs():
    if (airline == "Select" or source == "Select" or destination == "Select" or
        travel_class == "Select" or departure == "Select" or arrival == "Select"):
        st.warning("⚠️ Please select all flight details before predicting.")
        return False
    if source == destination:
        st.error("❌ Departure and Arrival cities are the same — please change one.")
        return False
    return True

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Search Fare"):
    if validate_inputs():
        try:
            stops_mapping = {
                "No Stop": "zero",
                "1 Stop": "one",
                "2 or More Stops": "two_or_more"
            }

            airline_encoded = le_dict['airline'].transform([airline.replace(" ", "_")])[0]

            input_data = {
                'airline': airline_encoded,
                'source_city': le_dict['source_city'].transform([source])[0],
                'departure_time': le_dict['departure_time'].transform([departure.replace(" ", "_")])[0],
                'stops': le_dict['stops'].transform([stops_mapping[stops]])[0],
                'arrival_time': le_dict['arrival_time'].transform([arrival.replace(" ", "_")])[0],
                'destination_city': le_dict['destination_city'].transform([destination])[0],
                'class': le_dict['class'].transform([travel_class])[0],
                'duration': duration,
                'days_left': days_left
            }

            input_df = pd.DataFrame([input_data])
            predicted_price = rf.predict(input_df)[0]

            st.markdown("---")
            st.markdown(
                f"<h3 style='text-align:center; color:#004c99;'>Estimated Fare: ₹{predicted_price:,.2f}</h3>",
                unsafe_allow_html=True
            )
            st.caption("*(Prediction based on airline, route, class, and travel time.)*")

        except Exception:
            st.error("⚠️ Something went wrong during prediction. Please recheck your inputs.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
Developed collaboratively by <b>Veeraj Thota</b> & <b>Sai Teja</b><br>
Machine Learning Project • GRIET 2025
</div>
""", unsafe_allow_html=True)
