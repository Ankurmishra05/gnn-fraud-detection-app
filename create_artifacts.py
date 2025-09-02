# create_artifacts.py

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("Starting artifact creation process...")

# --- 1. Load the original dataset ---
# Make sure 'creditcard.csv' is in your 'gnn_fraud_app' folder
try:
    df = pd.read_csv('creditcard.csv')
    print("Dataset 'creditcard.csv' loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'creditcard.csv' not found. Please make sure it's in the same folder.")
    exit()

# --- 2. Create and Save the two Scalers ---
print("Creating and saving amount_scaler.pkl...")
amount_scaler = StandardScaler()
amount_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
with open('amount_scaler.pkl', 'wb') as f:
    pickle.dump(amount_scaler, f)

print("Creating and saving time_scaler.pkl...")
time_scaler = StandardScaler()
time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
with open('time_scaler.pkl', 'wb') as f:
    pickle.dump(time_scaler, f)

# --- 3. Create and Save the Feature Predictor Model ---
X = df[['Time', 'Amount']]
y = df.loc[:, 'V1':'V28']

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

print("Training the feature prediction model... (This will take a few minutes)")
feature_predictor_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
feature_predictor_model.fit(X_scaled, y)
print("Feature prediction model trained successfully!")

with open('feature_predictor_model.pkl', 'wb') as f:
    pickle.dump(feature_predictor_model, f)

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)

print("\nAll new .pkl files have been created successfully using your local scikit-learn version!")