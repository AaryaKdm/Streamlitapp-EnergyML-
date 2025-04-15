import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
import os

if st.button('Back'):
    st.markdown(
        '''
        <meta http-equiv="refresh" content="0; url=http://127.0.0.1:5500/index.html" />
        ''', unsafe_allow_html=True)
st.image("imgs\streamlitlogo.png", use_container_width=True)

st.title("Future Energy Consumption Prediction")

# Add a sidebar for model information
st.sidebar.header("Model Information")

# Define default features used in the model training script
DEFAULT_FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
TARGET = 'PJME_MW'  # Default target from training script

# Load the model
model_path = "model.json"
model_loaded = False
reg = None
expected_features = []

# Attempt to load model with different methods
try:
    # First, try to load as pure XGBoost JSON model (used in training script)
    st.sidebar.info("Attempting to load XGBoost model from JSON...")
    reg = xgb.XGBRegressor()
    reg.load_model(model_path)
    model_loaded = True
    
    # Get feature names from model if possible
    if hasattr(reg, 'feature_names'):
        expected_features = reg.feature_names
    elif hasattr(reg, 'feature_names_in_'):
        expected_features = reg.feature_names_in_
    else:
        # Use default features from training script
        expected_features = DEFAULT_FEATURES
        st.sidebar.info("Using default features from training script.")
    
    st.sidebar.success("XGBoost model loaded successfully!")
    
except Exception as e:
    st.sidebar.warning(f"Could not load as direct XGBoost model: {str(e)}")
    
    # Try to load as custom JSON format
    try:
        with open(model_path, "r") as f:
            model_data = json.load(f)
        
        # Check if this is a metadata file pointing to another model file
        if "model_file" in model_data:
            actual_model_path = model_data["model_file"]
            st.sidebar.info(f"Model JSON references external model file: {actual_model_path}")
            
            # Try to load the referenced model file
            if os.path.exists(actual_model_path):
                file_ext = os.path.splitext(actual_model_path)[1].lower()
                
                if file_ext == ".pkl" or file_ext == ".pickle":
                    with open(actual_model_path, "rb") as f:
                        reg = pickle.load(f)
                    model_loaded = True
                    st.sidebar.success(f"Loaded model from {actual_model_path}")
                    
                elif file_ext == ".json":
                    reg = xgb.XGBRegressor()
                    reg.load_model(actual_model_path)
                    model_loaded = True
                    st.sidebar.success(f"Loaded XGBoost model from {actual_model_path}")
                    
                else:
                    st.sidebar.error(f"Unsupported model file format: {file_ext}")
            else:
                st.sidebar.error(f"Referenced model file not found: {actual_model_path}")
        
        # If we have features in the JSON, store them
        if "features" in model_data:
            expected_features = model_data["features"]
        elif "feature_names" in model_data:
            expected_features = model_data["feature_names"]
        
        # If model parameters are included, try to use them
        if not model_loaded and "model_type" in model_data:
            model_type = model_data["model_type"]
            model_params = model_data.get("model_params", {})
            
            if model_type == "xgboost":
                reg = xgb.XGBRegressor(**model_params)
                model_loaded = True
                st.sidebar.success("Created XGBoost model from parameters.")
            
            elif model_type == "random_forest":
                reg = RandomForestRegressor(**model_params)
                model_loaded = True
                st.sidebar.success("Created RandomForest model from parameters.")
            
            elif model_type == "gradient_boosting":
                reg = GradientBoostingRegressor(**model_params)
                model_loaded = True
                st.sidebar.success("Created GradientBoosting model from parameters.")
            
            else:
                st.sidebar.warning(f"Unsupported model type in JSON: {model_type}")
                
    except Exception as e:
        st.sidebar.error(f"Failed to load model as JSON: {str(e)}")

# If no model was loaded, try to load as pickle
if not model_loaded:
    try:
        st.sidebar.info("Attempting to load pickle model...")
        pickle_path = "xgboost_model.pkl"  # Fall back to original model path
        with open(pickle_path, "rb") as f:
            reg = pickle.load(f)
        model_loaded = True
        
        # Get feature names from model
        if hasattr(reg, 'feature_names_in_'):
            expected_features = reg.feature_names_in_
        
        st.sidebar.success(f"Model loaded from {pickle_path}!")
        
    except Exception as e:
        st.sidebar.error(f"Failed to load pickle model: {str(e)}")

# If we still don't have a model, stop
if not model_loaded:
    st.error("Failed to load model with any method. Please check the model file format.")
    st.stop()

# If we still don't have features, use defaults from training script
if len(expected_features) == 0:
    expected_features = DEFAULT_FEATURES
    st.sidebar.info("Using default features from original training script.")

# Display the feature names we found
st.sidebar.write(f"Model features ({len(expected_features)}):")
st.sidebar.write(", ".join([str(f) for f in expected_features]))

# Function to create time-based features matching the training script
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Function to add lag features matching the training script
def add_lags(df, target_col):
    # Create a dictionary mapping dates to target values
    target_map = df[target_col].to_dict()
    
    # Add lag features based on days back (matching training script)
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    
    # Fill NaN values (for dates without historical data)
    lag_cols = ['lag1', 'lag2', 'lag3']
    for col in lag_cols:
        if col in df.columns:
            # Use median of available values as fallback
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0
            df[col] = df[col].fillna(median_value)
    
    return df

# File uploader
uploaded_file = st.file_uploader("Upload your energy consumption dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file)
    
    # Ensure the datetime column is correctly formatted
    datetime_col = None
    if 'Datetime' in df.columns:
        datetime_col = 'Datetime'
    elif 'datetime' in df.columns:
        datetime_col = 'datetime'
    elif 'timestamp' in df.columns:
        datetime_col = 'timestamp'
    elif 'date' in df.columns:
        datetime_col = 'date'
    else:
        # Try to identify a datetime column
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                datetime_col = col
                break
            except:
                continue
    
    if datetime_col:
        df = df.set_index(datetime_col)
        df.index = pd.to_datetime(df.index)
    else:
        st.warning("No datetime column detected. Using row index instead.")
        df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    
    # Display Data Preview
    st.subheader("Data Preview")
    st.write(df.tail())
    
    # Identify the target column (energy consumption)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Fixed: Add a check to handle empty numeric_cols
    if not numeric_cols:
        st.error("No numeric columns found in the dataset. Please upload a dataset with numeric data.")
        st.stop()
    
    target_col = st.selectbox("Select the energy consumption column:", numeric_cols, 
                            index=0 if TARGET not in numeric_cols else numeric_cols.index(TARGET))
    
    # Apply the same feature engineering as in the training script
    st.info("Creating features to match model training...")
    df = create_features(df)
    df = add_lags(df, target_col)
    
    # Number of future days to predict
    future_days = st.slider("Select number of future days for prediction:", 1, 365, 30)
    
    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            # Get the exact last timestamp from the historical data
            last_timestamp = df.index.max()
            
            # Generate future timestamps starting EXACTLY after the last historical timestamp
            # Ensure there's no gap or overlap between historical and future data
            future_start = last_timestamp + pd.Timedelta(hours=1)
            future_dates = pd.date_range(start=future_start, 
                                         periods=future_days * 24, freq="H")
            
            # Create future DataFrame with same format as training data
            future_df = pd.DataFrame(index=future_dates)
            future_df[target_col] = np.nan  # Placeholder for the target
            
            # Combine historical and future data for feature creation
            combined_df = pd.concat([df[[target_col]], future_df])
            
            # Create features for the combined dataset
            combined_df = create_features(combined_df)
            combined_df = add_lags(combined_df, target_col)
            
            # Isolate future data with features - make sure we only get the future dates
            future_with_features = combined_df.loc[future_dates]
            
            # Check if expected_features are all present and handle any missing ones
            missing_features = [f for f in expected_features if f not in future_with_features.columns]
            if missing_features:
                st.warning(f"Missing features: {missing_features}. Adding them with default values.")
                for feat in missing_features:
                    future_with_features[feat] = 0  # Add missing features with default value
            
            # Select only the features required by the model
            X_future = future_with_features[expected_features]
            
            # Show what we're predicting with
            with st.expander("View prediction features"):
                st.write(X_future.head())
            
            # Make predictions
            try:
                predictions = reg.predict(X_future)
                future_with_features['predicted'] = predictions
                
                # Display results
                st.subheader("Future Predictions")
                st.write(future_with_features[['predicted']].head(10))
                
                # Create visualization
                st.subheader("Future Energy Consumption Forecast")
                fig, ax = plt.subplots(figsize=(12, 5))
                
                # Plot historical data (last 7 days for context)
                historical_days = min(7, len(df) // 24) if len(df) > 24 else 1
                historical_period = df.index[-historical_days*24:]
                historical_values = df[target_col][-historical_days*24:]
                
                # Plot historical data
                ax.plot(historical_period, historical_values, 
                        label="Historical Energy Consumption", color="blue")
                
                # Plot predictions - starting exactly where historical data ends
                ax.plot(future_with_features.index, predictions, 
                        label="Predicted Energy Consumption", linestyle="dashed", color="red")
                
                # Add vertical line to clearly separate historical from predicted data
                ax.axvline(x=last_timestamp, color='green', linestyle='--', 
                          label="Prediction Start")
                
                # Add text annotation for clarity
                ax.text(last_timestamp, ax.get_ylim()[1] * 0.95, 
                        " Predictions start", verticalalignment='top')
                
                # Highlight the transition area
                ax.axvspan(last_timestamp - pd.Timedelta(hours=12), 
                           last_timestamp + pd.Timedelta(hours=12),
                           alpha=0.2, color='yellow')
                
                ax.legend()
                ax.set_title("Energy Consumption Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Energy Consumption")
                
                # Add a grid for better readability
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Adjust x-axis to ensure the boundary is clear
                ax.set_xlim(historical_period[0], future_dates[-1])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show exact transition point
                st.info(f"**Prediction boundary**: Historical data ends at {last_timestamp}, " 
                        f"predictions start at {future_start}")
                
                # Download button
                csv_data = future_with_features[['predicted']].rename(
                    columns={'predicted': f'Predicted_{target_col}'})
                
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data.to_csv().encode("utf-8"),
                    file_name="future_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                st.write("Debug information:")
                st.write(f"Feature columns available: {list(X_future.columns)}")
                st.write(f"Features expected: {list(expected_features)}")
                st.write(f"Missing features: {[f for f in expected_features if f not in X_future.columns]}")
                
                # Check for NaN values
                nan_columns = X_future.columns[X_future.isna().any()].tolist()
                if nan_columns:
                    st.write("Warning: NaN values found in these columns:")
                    st.write(nan_columns)
                    st.write("NaN counts:")
                    st.write(X_future[nan_columns].isna().sum())