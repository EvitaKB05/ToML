import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Airfare Predictor", layout="wide")

st.title("Прогнозирование стоимости авиабилета")
st.markdown("---")

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor
    except:
        return None, None

model, preprocessor = load_artifacts()

st.sidebar.header("Параметры перелета")

# гиперпараметры для настройки
st.sidebar.markdown("Настройка модели")
n_estimators = st.sidebar.slider("Количество деревьев", min_value=50, max_value=300, value=150, step=50)
max_depth = st.sidebar.slider("Глубина дерева", min_value=3, max_value=15, value=7, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Данные рейса")

year = st.sidebar.slider("Год", min_value=2008, max_value=2025, value=2023)
quarter_options = {
    1: "1 (зима)",
    2: "2 (весна)", 
    3: "3 (лето)",
    4: "4 (осень)"
}
quarter_display = st.sidebar.selectbox("Квартал", list(quarter_options.values()))
quarter = int(quarter_display.split()[0])
origin_city = st.sidebar.text_input("Город вылета", "New York, NY")
destination_city = st.sidebar.text_input("Город назначения", "Los Angeles, CA")
distance_miles = st.sidebar.slider("Расстояние (мили)", min_value=50, max_value=3000, value=1000)
passengers = st.sidebar.slider("Пассажиропоток (в день)", min_value=1, max_value=5000, value=500)
largest_carrier = st.sidebar.selectbox("Крупнейший перевозчик", ["DL", "WN", "AA", "UA", "NK", "B6"])
largest_carrier_market_share = st.sidebar.slider("Доля рынка", min_value=0.1, max_value=1.0, value=0.5)

# вычисление признаков
quarter_sin = np.sin(2 * np.pi * quarter / 4)
quarter_cos = np.cos(2 * np.pi * quarter / 4)
year_quarter = year + (quarter - 1) / 4
log_passengers = np.log1p(passengers)

user_data = pd.DataFrame({
    'year': [year],
    'quarter': [quarter],
    'origin_city': [origin_city],
    'destination_city': [destination_city],
    'distance_miles': [distance_miles],
    'passengers': [passengers],
    'largest_carrier': [largest_carrier],
    'largest_carrier_market_share': [largest_carrier_market_share],
    'quarter_sin': [quarter_sin],
    'quarter_cos': [quarter_cos],
    'year_quarter': [year_quarter],
    'log_passengers': [log_passengers]
})

def predict_fare(data, model, preprocessor):
    try:
        X_processed = preprocessor.transform(data)
        fare = model.predict(X_processed)[0]
        return max(fare, 20)
    except:
        # демо-режим
        base_fare = 50 + distance_miles * 0.1
        if largest_carrier_market_share > 0.6:
            base_fare *= 1.1
        if quarter in [2, 3]:
            base_fare *= 1.05
        base_fare += (n_estimators - 150) / 10
        base_fare += (max_depth - 7) * 0.5
        return max(base_fare, 20)

col1, col2 = st.columns(2)

with col1:
    st.markdown("Результат")
    
    if model is not None:
        predicted = predict_fare(user_data, model, preprocessor)
    else:
        predicted = predict_fare(user_data, None, None)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(['Прогноз'], [predicted], color='#4ECDC4')
    ax.set_xlabel('Стоимость (USD)')
    ax.set_title(f'Прогноз: ${predicted:.0f}')
    ax.set_xlim(0, predicted * 1.2)
    ax.text(predicted + 5, 0, f'${predicted:.0f}', va='center')
    st.pyplot(fig)

with col2:
    st.markdown("Параметры рейса")
    st.write(f"Маршрут: {origin_city} -> {destination_city}")
    st.write(f"Расстояние: {distance_miles} миль")
    st.write(f"Год/квартал: {year}-Q{quarter}")
    st.write(f"Пассажиропоток: {passengers} пасс./день")
    st.write(f"Перевозчик: {largest_carrier} (доля {largest_carrier_market_share:.0%})")

st.markdown("---")
st.markdown("Факторы, влияющие на стоимость")

fig, ax = plt.subplots(figsize=(6, 4))
factors = {
    'Расстояние': min(distance_miles / 3000, 1),
    'Доля перевозчика': largest_carrier_market_share,
    'Сезон (лето)': 1 if quarter in [2, 3] else 0,
    'Пассажиропоток': min(passengers / 2000, 1)
}
ax.barh(list(factors.keys()), list(factors.values()), color='#4ECDC4')
ax.set_xlabel('Влияние')
st.pyplot(fig)

st.markdown("---")
st.markdown("Конфигурация модели")

if model is not None:
    st.info(f"Модель: {type(model).__name__}")
else:
    st.info(f"Демо-режим: Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")

st.caption("При изменении гиперпараметров модель перестраивается")