import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Use scikeras for the KerasRegressor wrapper
from scikeras.wrappers import KerasRegressor

# Import Keras from the standalone package
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
# 1) Data Preprocessing and Feature Engineering
###############################################################################
def preprocess_data(df):
    """
    Drops unnecessary columns, converts types, maps strings to numbers,
    and creates an interaction feature.
    """
    # Drop unnecessary columns
    cols_to_drop = ["DetectDate", "LeakFix (Date)", "Unnamed: 18"]
    df = df.drop(columns=cols_to_drop, errors="ignore").copy()
    
    # Convert selected columns to string
    df["Veg Damage"] = df["Veg Damage"].astype(str)
    df["Equipment"] = df["Equipment"].astype(str)
    
    # Map 'Spread' to numeric (in feet)
    spread_map = {
        "0 - 5'": 2.5, 
        "6 - 10'": 8, 
        "11 - 20'": 15, 
        "21 - 40'": 30, 
        "> 40'": 45, 
        "Unknown": np.nan
    }
    df["Spread_ft"] = df["Spread"].map(spread_map)
    
    # Map 'GasPercentage' to numeric values
    gas_map = {
        "0%": 0, 
        "1 - 3%": 2, 
        "4 - 10%": 7, 
        "11 - 20%": 15, 
        "21 - 40%": 30, 
        "> 40%": 45, 
        "Unknown": np.nan
    }
    df["GasPct_num"] = df["GasPercentage"].map(gas_map)
    
    # Parse and convert 'PipeSize' to numeric if available
    def parse_pipe_size(val):
        if pd.isna(val) or val == "Unknown":
            return np.nan
        cleaned = str(val).replace('"', '').replace(" inch", "").strip()
        try:
            return float(eval(cleaned))
        except Exception:
            return np.nan
    if "PipeSize" in df.columns:
        df["PipeSize_num"] = df["PipeSize"].apply(parse_pipe_size)
    
    # Map 'Pressure' to numeric values
    pressure_map = {"Low Pressure": 1, "Medium Pressure": 2, "High Pressure": 3, "Unknown": np.nan}
    if "Pressure" in df.columns:
        df["Pressure_num"] = df["Pressure"].map(pressure_map)
    
    # Create an interaction feature
    df["Spread_Gas_Interaction"] = df["Spread_ft"] * df["GasPct_num"]
    
    return df

###############################################################################
# 2) Load and Clean Data
###############################################################################
def load_data(filepath, cap_value=50000):
    """
    Loads data from an Excel file, cleans it, caps the target, fills missing values,
    and applies feature engineering.
    """
    df = pd.read_excel(filepath)
    
    # Convert 'Date' to int and sort
    df["Date"] = df["Date"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Clean and cap the 'AMOUNT' column
    df["AMOUNT"] = df["AMOUNT"].replace(r'[\$,]', '', regex=True).astype(float)
    df["AMOUNT"] = np.minimum(df["AMOUNT"], cap_value)
    
    # Fill missing categorical values with "Unknown"
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
    # Fill missing numeric values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    df = preprocess_data(df)
    return df

###############################################################################
# 3) Bin Splitting (7 Bins)
###############################################################################
def split_into_bins(df):
    """
    Splits the dataset into 7 bins based on the 'AMOUNT' ranges.
    """
    bins = [0, 500, 1000, 2000, 5000, 10000, 20000, np.inf]
    labels = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5", "Bin6", "Bin7"]
    df = df.copy()
    df["Bin"] = pd.cut(df["AMOUNT"], bins=bins, labels=labels, right=True)
    df = df.dropna(subset=["Bin"])
    return df

###############################################################################
# 4) Create Preprocessor (for Bins 2-7)
###############################################################################
def create_preprocessor(X):
    """
    Builds a preprocessing pipeline with imputation, scaling, and polynomial interactions.
    Used for Bins 2–7.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    return preprocessor

###############################################################################
# 5) Build Neural Network Model
###############################################################################
def build_model(input_shape=(1,), num_layers=2, num_units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds and compiles a simple feedforward neural network using standalone Keras.
    The input_shape is expected as a tuple (n_features,). A default value is provided.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    for _ in range(num_layers):
        model.add(Dense(num_units, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

###############################################################################
# 6) Train Neural Network Model per Bin
###############################################################################
def train_model_nn(X, y, bin_label=""):
    """
    Trains a neural network model for the given bin.
    
    - For Bin1, uses a simpler preprocessor (no polynomial features).
    - For Bins 2–7, uses a richer preprocessor with polynomial interaction features.
    
    The neural network model is wrapped in a scikeras KerasRegressor.
    """
    # Choose preprocessor based on bin
    if bin_label == "Bin1":
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object"]).columns
        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])
    else:
        preprocessor = create_preprocessor(X)
    
    # Create a pipeline: Preprocessor -> Neural Network
    nn_regressor = KerasRegressor(
        model=build_model,
        # Hyperparameters (adjust as needed):
        epochs=100,
        batch_size=32,
        verbose=0,
        num_layers=2,
        num_units=64,
        dropout_rate=0.2,
        learning_rate=0.001,
        # Early stopping to prevent overfitting
        fit__callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("nn", nn_regressor)
    ])
    
    pipeline.fit(X, y)
    return pipeline

###############################################################################
# 7) Calculate Metrics
###############################################################################
def calculate_metrics(y_true, y_pred, tolerance=0.2):
    """
    Computes:
      - Percentage of predictions within ±20% of the true value (custom accuracy)
      - Mean Absolute Error (MAE)
      - Root Mean Squared Error (RMSE)
    """
    within_tolerance = np.abs(y_pred - y_true) <= tolerance * y_true
    accuracy = np.mean(within_tolerance) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return accuracy, mae, rmse

###############################################################################
# 8) Main Execution
###############################################################################
def main():
    # Update this filepath to point to your Excel data file
    filepath = "LeakRepairData.xlsx"
    df = load_data(filepath)
    
    # Bin the data and display bin counts
    df_binned = split_into_bins(df)
    print("Bin Distribution:")
    print(df_binned["Bin"].value_counts())
    
    # Define training and testing years
    train_years = [2021, 2022, 2023]
    test_year = 2024
    
    # Reload full data for splitting by year
    df_all = load_data(filepath)
    df_train = df_all[df_all["Date"].isin(train_years)]
    df_test = df_all[df_all["Date"] == test_year]
    
    df_train = split_into_bins(df_train)
    df_test = split_into_bins(df_test)
    
    bin_order = ["Bin1", "Bin2", "Bin3", "Bin4", "Bin5", "Bin6", "Bin7"]
    
    for bin_label in bin_order:
        print(f"\nProcessing {bin_label}...")
        train_bin = df_train[df_train["Bin"] == bin_label]
        test_bin = df_test[df_test["Bin"] == bin_label]
        if test_bin.empty:
            print(f"[{bin_label}] No test data available.")
            continue
        X_train = train_bin.drop(columns=["AMOUNT", "Bin"])
        y_train = train_bin["AMOUNT"]
        X_test = test_bin.drop(columns=["AMOUNT", "Bin"])
        y_test = test_bin["AMOUNT"]
        
        # Train the neural network model for this bin
        model = train_model_nn(X_train, y_train, bin_label=bin_label)
        
        # Make predictions and compute metrics
        y_pred = model.predict(X_test)
        accuracy, mae, rmse = calculate_metrics(y_test, y_pred)
        print(f"[{bin_label}] ±20% Accuracy: {accuracy:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        
        # (Optional) Diagnostic plot: Uncomment to see predicted vs. actual scatter plot.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual AMOUNT")
        plt.ylabel("Predicted AMOUNT")
        plt.title(f"{bin_label}: Actual vs. Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()
        """
    
if __name__ == "__main__":
    main()
