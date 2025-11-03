#  Airline Fare Prediction

A machine learning–powered web application that predicts the flight fare based on airline, route, class, travel time, and days left before departure.  
Built collaboratively by **Veeraj Thota** and **Sai Teja**, this project combines data preprocessing, model comparison, and a modern Streamlit interface for real-time fare estimation.

---

##  Overview

The **Airline Fare Prediction System** leverages regression models to estimate ticket prices across multiple Indian airlines.  
It provides a simple, interactive UI where users can select their travel details and instantly view the predicted fare.

This project aims to help travelers and analysts understand how different factors — such as airline, class, stops, and booking time — affect airfare trends.

---

##  Features

-  **Machine Learning Models:** Linear Regression and Random Forest Regressor  
-  **Performance Comparison:** Evaluates R² Score, MAE, and RMSE  
-  **Interactive Interface:** Built using Streamlit for real-time predictions  
-  **Data Preprocessing:** Automatic encoding for categorical variables  
-  **Error Handling:** Prevents invalid inputs and duplicate city selection  
-  **Clean UI:** Simple, airline-style form design inspired by real booking websites  

---

##  Dataset

The model was trained using `airlines_flights_data.csv`, which includes:

| Feature | Description |
|----------|-------------|
| Airline | Airline name (e.g., Indigo, Vistara, Air India) |
| Source City | Departure location |
| Destination City | Arrival location |
| Departure Time | Time slot of departure (Morning, Evening, etc.) |
| Arrival Time | Time slot of arrival |
| Stops | Number of stops (No Stop, 1 Stop, 2+ Stops) |
| Class | Economy or Business |
| Duration | Total travel duration (hours) |
| Days Left | Days before departure |
| Price | Target variable (fare in ₹) |

---

##  Installation and Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/vijay12968/airline-fare-prediction.git
cd airline-fare-prediction
pip install -r requirements.txt

```

To run the app locally:

```bash
streamlit run app.py
```
