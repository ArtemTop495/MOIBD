import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv('real_estate_data.csv')

# Конвертация цены в TRY
data['price'] = pd.to_numeric(data['price'], errors='coerce')
def convert_to_try(row):
    try:
        price = row['price']
        if pd.isna(price):
            return np.nan
        rates = {'USD': 41.15, 'EUR': 48.17, 'GBP': 55.60}
        return price * rates.get(row['price_currency'], 1)  # 1 для TRY или unknown
    except:
        return np.nan

data['price_try'] = data.apply(convert_to_try, axis=1)
data = data.dropna(subset=['price_try'])  # Удаляем NaN в цене

# Обработка адреса: разбиваем на city, district, neighborhood
data['city'] = data['address'].str.split('/').str[0].fillna('Unknown')
data['district'] = data['address'].str.split('/').str[1].fillna('Unknown')
data['neighborhood'] = data['address'].str.split('/').str[2].fillna('Unknown')

# Кодируем city, district, neighborhood с помощью LabelEncoder (как listing_type)
le_city = LabelEncoder()
data['city'] = le_city.fit_transform(data['city'].astype(str))

le_district = LabelEncoder()
data['district'] = le_district.fit_transform(data['district'].astype(str))

le_neighborhood = LabelEncoder()
data['neighborhood'] = le_neighborhood.fit_transform(data['neighborhood'].astype(str))

# Удаляем address
data = data.drop(['address'], axis=1)

# Обработка size: заполняем пропуски медианой
data['size'] = data['size'].replace('', np.nan).astype(float)
data['size'].fillna(data['size'].median(), inplace=True)

# Обработка building_age
data['building_age'] = data['building_age'].replace('0', '0-5 arası').replace('40 ve üzeri', '40').replace('', np.nan)
data['building_age'] = data['building_age'].apply(lambda x: 0 if pd.isna(x) else int(x.split('-')[0]) if 'arası' in str(x) else int(x))

# Обработка total_floor_count
data['total_floor_count'] = data['total_floor_count'].replace('20 ve üzeri', '20').replace('', np.nan)
data['total_floor_count'] = data['total_floor_count'].apply(lambda x: np.nan if pd.isna(x) else float(x.split('-')[0]) if 'arası' in str(x) else float(x))
data['total_floor_count'].fillna(data['total_floor_count'].median(), inplace=True)

# Обработка floor_no
floor_mapping = {
    'Zemin Kat': 0, 'Giriş Katı': 0, 'Yüksek Giriş': 1, 'Bahçe katı': 0,
    'Kot 2': -2, 'Kot1': -1, 'Asma Kat': 0.5, 'Çatı Katı': 100, 'En Üst Kat': 100,
    '20 ve üzeri': 20
}
data['floor_no'] = data['floor_no'].replace(floor_mapping).replace('', np.nan)
data['floor_no'] = pd.to_numeric(data['floor_no'], errors='coerce')
data['floor_no'].fillna(data['floor_no'].median(), inplace=True)

# Обработка room_count
def parse_room_count(x):
    if pd.isna(x) or x == '' or x == '+':
        return np.nan
    try:
        parts = x.split('+')
        return sum(int(part) for part in parts if part.strip().isdigit())
    except:
        return np.nan

data['room_count'] = data['room_count'].apply(parse_room_count)
data['room_count'].fillna(data['room_count'].median(), inplace=True)

# Дополнительные фичи
data['price_per_m2'] = data['price_try'] / data['size'].clip(lower=1)  # Избегаем деления на 0
data['age_size_interact'] = data['building_age'] * data['size']  # Взаимодействие

# Фильтрация выбросов
data = data[(data['price_try'] > 1000) & (data['price_try'] < 1e9)]
data = data[(data['size'] > 10) & (data['size'] < 5000)]

# Кодируем категориальные переменные
categorical_cols = ['type', 'sub_type', 'heating_type']
for col in categorical_cols:
    data[col] = data[col].replace('', np.nan).fillna('Unknown')
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Удаляем ненужные колонки
data = data.drop(['id', 'start_date', 'end_date', 'furnished', 'price', 'price_currency'], axis=1, errors='ignore')

# Матрица корреляций
corr_matrix = data.drop(columns=['type']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Разделяем на X и y
X = data.drop('price_try', axis=1)
y = data['price_try']

# Разделяем данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, alpha=1, tree_method='hist', random_state=42)
stacking_model = StackingRegressor(
    estimators=[('dt', DecisionTreeRegressor(max_depth=10, random_state=42)), ('ridge', Ridge(alpha=1.0))],
    final_estimator=LinearRegression(), cv=5
)

models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Stacking': stacking_model
}

# Тюнинг XGBoost
param_grid = {'learning_rate': [0.05, 0.1], 'max_depth': [5, 7], 'n_estimators': [100, 200]}
grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)
xgb_model = grid.best_estimator_
print("Best params for XGBoost:", grid.best_params_)

# Обучение и оценка
results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    predictions[name] = y_pred
    print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

# Лучшая модель
best_model = max(results.items(), key=lambda x: x[1]['R2'])  # Выбираем по R²
print(f"\nBest Model: {best_model[0]} with MSE: {best_model[1]['MSE']:.2f} and R2: {best_model[1]['R2']:.2f}")

# Визуализация 1: Bar Plot для MSE и R²
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=list(results.keys()), y=[results[model]['MSE'] for model in results], ax=ax1)
ax1.set_title('Mean Squared Error (MSE) Comparison')
ax1.set_ylabel('MSE')
ax1.tick_params(axis='x', rotation=45)
sns.barplot(x=list(results.keys()), y=[results[model]['R2'] for model in results], ax=ax2)
ax2.set_title('R² Score Comparison')
ax2.set_ylabel('R² Score')
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Визуализация 2: Scatter Plots для предсказанных vs реальных цен
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Predicted vs Actual Prices')
for i, (name, y_pred) in enumerate(predictions.items()):
    axes[i].scatter(y_test, y_pred, alpha=0.5)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_title(name)
    axes[i].set_xlabel('Actual Price (TRY)')
    axes[i].set_ylabel('Predicted Price (TRY)')
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
plt.tight_layout()
plt.show()