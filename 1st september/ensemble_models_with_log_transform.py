import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('real_estate_data.csv')

# Inspect price and price_currency for issues
print("Initial price null count:", data['price'].isnull().sum())
print("Initial price_currency null count:", data['price_currency'].isnull().sum())
print("Unique price_currency values:", data['price_currency'].unique())

# Convert price to numeric, handle invalid values
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# Currency conversion to TRY (rates as of September 1, 2025)
def convert_to_try(row):
    try:
        price = row['price']
        if pd.isna(price):
            return np.nan
        if row['price_currency'] == 'USD':
            return price * 41.15
        elif row['price_currency'] == 'EUR':
            return price * 48.17
        elif row['price_currency'] == 'GBP':
            return price * 55.60
        elif row['price_currency'] in ['TRY', '', np.nan]:
            return price
        else:
            print(f"Warning: Unrecognized currency '{row['price_currency']}' for row {row.name}")
            return price
    except:
        return np.nan

data['price_try'] = data.apply(convert_to_try, axis=1)

# Drop rows with NaN or non-positive prices
print("Rows with NaN in price_try before drop:", data['price_try'].isnull().sum())
print("Rows with price_try <= 0 before drop:", (data['price_try'] <= 0).sum())
data = data[data['price_try'] > 0]  # Drop NaN and non-positive prices
print("Rows with NaN in price_try after drop:", data['price_try'].isnull().sum())
print("Rows with price_try <= 0 after drop:", (data['price_try'] <= 0).sum())

# Separate sale and rent data
sale_data = data[data['listing_type'] == 1].copy()
rent_data = data[data['listing_type'] == 2].copy()
print(f"Sale data rows: {len(sale_data)}, Rent data rows: {len(rent_data)}")

# Function to preprocess data
def preprocess_data(df):
    # Drop irrelevant columns (including listing_type since it's constant)
    df = df.drop(['id', 'start_date', 'end_date', 'address', 'price', 'price_currency', 'listing_type'], axis=1)

    # Handle missing values
    df['size'] = df['size'].replace('', np.nan).astype(float)
    df['size'].fillna(df['size'].median(), inplace=True)

    # Convert building_age to numeric, handling ranges and '40 ve üzeri'
    df['building_age'] = df['building_age'].replace('0', '0-5 arası').replace('40 ve üzeri', '40').replace('', np.nan)
    df['building_age'] = df['building_age'].apply(
        lambda x: 0 if pd.isna(x) else int(x.split('-')[0]) if 'arası' in str(x) else int(x)
    )

    # Convert total_floor_count to numeric, handling ranges and '20 ve üzeri'
    df['total_floor_count'] = df['total_floor_count'].replace('20 ve üzeri', '20').replace('', np.nan)
    df['total_floor_count'] = df['total_floor_count'].apply(
        lambda x: np.nan if pd.isna(x) else float(x.split('-')[0]) if 'arası' in str(x) else float(x)
    )
    df['total_floor_count'].fillna(df['total_floor_count'].median(), inplace=True)

    # Convert floor_no to numeric based on data dictionary
    floor_mapping = {
        'Zemin Kat': 0, 'Giriş Katı': 0, 'Yüksek Giriş': 1, 'Bahçe katı': 0,
        'Kot 2': -2, 'Kot1': -1, 'Asma Kat': 0.5, 'Çatı Katı': 100, 'En Üst Kat': 100,
        '20 ve üzeri': 20
    }
    df['floor_no'] = df['floor_no'].replace(floor_mapping).replace('', np.nan)
    df['floor_no'] = pd.to_numeric(df['floor_no'], errors='coerce')
    df['floor_no'].fillna(df['floor_no'].median(), inplace=True)

    # Convert room_count to numeric by summing rooms and living rooms
    def parse_room_count(x):
        if pd.isna(x) or x == '' or x == '+':
            return np.nan
        try:
            parts = x.split('+')
            return sum(int(part) for part in parts if part.strip().isdigit())
        except:
            return np.nan

    df['room_count'] = df['room_count'].apply(parse_room_count)
    df['room_count'].fillna(df['room_count'].median(), inplace=True)

    # Encode categorical variables
    categorical_cols = ['type', 'sub_type', 'heating_type', 'furnished']
    for col in categorical_cols:
        df[col] = df[col].replace('', np.nan).fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# Preprocess sale and rent data
sale_data = preprocess_data(sale_data)
rent_data = preprocess_data(rent_data)

# Define features and target for sale and rent
sale_X = sale_data.drop('price_try', axis=1)
sale_y = sale_data['price_try']
sale_y_log = np.log1p(sale_y)

rent_X = rent_data.drop('price_try', axis=1)
rent_y = rent_data['price_try']
rent_y_log = np.log1p(rent_y)

# Check for NaN in features and target
print("Sale data - NaN in X:\n", sale_X.isnull().sum())
print("Sale data - NaN in y_log:", sale_y_log.isnull().sum())
print("Rent data - NaN in X:\n", rent_X.isnull().sum())
print("Rent data - NaN in y_log:", rent_y_log.isnull().sum())

# Split data
sale_X_train, sale_X_test, sale_y_train, sale_y_test = train_test_split(sale_X, sale_y_log, test_size=0.2, random_state=42)
rent_X_train, rent_X_test, rent_y_train, rent_y_test = train_test_split(rent_X, rent_y_log, test_size=0.2, random_state=42)

# Initialize models
def initialize_models():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting (XGBoost)': XGBRegressor(n_estimators=100, random_state=42),
        'Stacking': StackingRegressor(
            estimators=[
                ('dt', DecisionTreeRegressor(random_state=42)),
                ('svr', SVR())
            ],
            final_estimator=LinearRegression()
        )
    }

# Train and evaluate models for sale and rent
def train_evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    models = initialize_models()
    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # Revert log transformation
        y_test_orig = np.expm1(y_test)  # Revert log transformation
        mse = mean_squared_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        predictions[name] = (y_pred, y_test_orig)
        print(f"{dataset_name} - {name} - MSE: {mse:.2f}, R2: {r2:.2f}")
    return results, predictions

# Train and evaluate for sale and rent
sale_results, sale_predictions = train_evaluate_models(sale_X_train, sale_X_test, sale_y_train, sale_y_test, "Sale")
rent_results, rent_predictions = train_evaluate_models(rent_X_train, rent_X_test, rent_y_train, rent_y_test, "Rent")

# Find best models
sale_best_model = min(sale_results.items(), key=lambda x: x[1]['MSE'])
rent_best_model = min(rent_results.items(), key=lambda x: x[1]['MSE'])
print(f"\nSale Best Model: {sale_best_model[0]} with MSE: {sale_best_model[1]['MSE']:.2f} and R2: {sale_best_model[1]['R2']:.2f}")
print(f"Rent Best Model: {rent_best_model[0]} with MSE: {rent_best_model[1]['MSE']:.2f} and R2: {rent_best_model[1]['R2']:.2f}")

# Visualization: Bar Plots for MSE and R2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Sale MSE
sns.barplot(x=list(sale_results.keys()), y=[sale_results[model]['MSE'] for model in sale_results], ax=ax1)
ax1.set_title('Sale: Mean Squared Error (MSE)')
ax1.set_ylabel('MSE')
ax1.tick_params(axis='x', rotation=45)

# Sale R2
sns.barplot(x=list(sale_results.keys()), y=[sale_results[model]['R2'] for model in sale_results], ax=ax2)
ax2.set_title('Sale: R² Score')
ax2.set_ylabel('R² Score')
ax2.tick_params(axis='x', rotation=45)

# Rent MSE
sns.barplot(x=list(rent_results.keys()), y=[rent_results[model]['MSE'] for model in rent_results], ax=ax3)
ax3.set_title('Rent: Mean Squared Error (MSE)')
ax3.set_ylabel('MSE')
ax3.tick_params(axis='x', rotation=45)

# Rent R2
sns.barplot(x=list(rent_results.keys()), y=[rent_results[model]['R2'] for model in rent_results], ax=ax4)
ax4.set_title('Rent: R² Score')
ax4.set_ylabel('R² Score')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Visualization: Scatter Plots for Predicted vs Actual Prices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Predicted vs Actual Prices')

# Sale Scatter Plots
for i, (name, (y_pred, y_test_orig)) in enumerate(sale_predictions.items()):
    axes[0, i].scatter(y_test_orig, y_pred, alpha=0.5)
    axes[0, i].plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    axes[0, i].set_title(f"Sale: {name}")
    axes[0, i].set_xlabel('Actual Price (TRY)')
    axes[0, i].set_ylabel('Predicted Price (TRY)')
    axes[0, i].set_xscale('log')
    axes[0, i].set_yscale('log')

# Rent Scatter Plots
for i, (name, (y_pred, y_test_orig)) in enumerate(rent_predictions.items()):
    axes[1, i].scatter(y_test_orig, y_pred, alpha=0.5)
    axes[1, i].plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    axes[1, i].set_title(f"Rent: {name}")
    axes[1, i].set_xlabel('Actual Price (TRY)')
    axes[1, i].set_ylabel('Predicted Price (TRY)')
    axes[1, i].set_xscale('log')
    axes[1, i].set_yscale('log')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()