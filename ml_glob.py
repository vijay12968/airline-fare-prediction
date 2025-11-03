# -*- coding: utf-8 -*-
"""Train and save both models for Airline Fare Prediction"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# -----------------------------------------
# Load Dataset
# -----------------------------------------
df = pd.read_csv("airlines_flights_data_small.csv")
df.drop(columns=['index', 'flight'], inplace=True)

# Encode Categorical Columns
cat_cols = ['airline', 'source_city', 'departure_time', 'stops',
            'arrival_time', 'destination_city', 'class']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Feature & Target Split
X = df.drop('price', axis=1)
y = df['price']

# Train-Test Split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------
# Train Linear Regression
# -----------------------------------------
linr = LinearRegression()
linr.fit(X_tr, y_tr)
pred_lin = linr.predict(X_val)

r2_lin = r2_score(y_val, pred_lin)
mae_lin = mean_absolute_error(y_val, pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_val, pred_lin))

print("🔹 Linear Regression Results:")
print(f"R²: {r2_lin*100:.2f}% | MAE: {mae_lin:.2f} | RMSE: {rmse_lin:.2f}")

# -----------------------------------------
# Train Random Forest
# -----------------------------------------
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_val)

r2_rf = r2_score(y_val, pred_rf)
mae_rf = mean_absolute_error(y_val, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))

print("\n🌳 Random Forest Results:")
print(f"R²: {r2_rf*100:.2f}% | MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}")

# -----------------------------------------
# Compare Models
# -----------------------------------------
print("\n📊 Comparison")
print(f"Linear Regression -> {r2_lin*100:.2f}% | Random Forest -> {r2_rf*100:.2f}%")

# -----------------------------------------
# Save Models and Encoders
# -----------------------------------------
pickle.dump(linr, open("linear_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))
pickle.dump(le_dict, open("encoders.pkl", "wb"))

print("\n✅ Models & encoders saved successfully.")

import pickle

# Save both models after training
with open("linear_model.pkl", "wb") as f:
    pickle.dump(linr, f)

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("✅ Models saved successfully as linear_model.pkl and random_forest_model.pkl")

