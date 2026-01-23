import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Оценка недвижимости", page_icon="🏠", layout="wide")

st.title("🏠 Калькулятор стоимости недвижимости")
st.markdown("### Модель Random Forest с улучшенной чувствительностью")

try:
    response = requests.get("http://localhost:8077/health", timeout=3)
    health_info = response.json()
    
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.success(f"✅ Модель активна")
        st.info(f"**Валюта:** {health_info['currency']}")
        st.info(f"**Курс:** 1 TRY = {health_info['exchange_rate']} RUB")

        st.subheader("📊 Чувствительность к площади")
        example_resp = requests.get("http://localhost:8077/calculate_price_example")
        if example_resp.status_code == 200:
            examples = example_resp.json()["examples"]
            for ex in examples:
                st.write(f"{ex['size_m2']} м² → {ex['predicted_price_rub']:,.0f} RUB")
                
except:
    st.sidebar.warning("⚠️ API не отвечает")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Основные параметры")

    with st.expander("📍 Местоположение", expanded=True):
        city = st.selectbox("Город", ["İstanbul", "Ankara", "İzmir", "Bursa"])
        district = st.text_input("Район", "Kadıköy")
        neighborhood = st.text_input("Микрорайон", "Moda")
    
    with st.expander("🏢 Характеристики здания", expanded=True):
        type_ = st.selectbox("Тип", ["Konut", "İş Yeri", "Arsa"])
        sub_type = st.selectbox("Подтип", ["Daire", "Villa", "Müstakil", "Residence"])
        building_age = st.selectbox("Возраст", ["0-5", "5-10", "10-20", "20-30", "30+"])
        total_floor_count = st.slider("Этажей в доме", 1, 50, 5)
        floor_no = st.selectbox("Этаж", ["1", "2", "3", "4", "5", "Giriş Kat", "Çatı Katı"])
    
    with st.expander("📐 Параметры квартиры", expanded=True):
        room_count = st.select_slider("Комнат", options=[1, 2, 3, 4, 5, 6], value=2)
        size = st.slider("Площадь (м²)", 30, 200, 65, 5)
        heating_type = st.selectbox("Отопление", [
            "Kalorifer (Doğalgaz)", "Kombi (Doğalgaz)", "Merkezi Sistem", "Yok"
        ])

with col2:
    st.subheader("Дополнительные параметры")
    
    listing_type = st.selectbox("Тип сделки", ["Satılık", "Kiralık"])
    tom = st.slider("Дней на рынке", 0, 365, 30)
    
    st.markdown("---")
    st.subheader("💰 Результат")

    if st.button("🎯 Рассчитать стоимость", type="primary", use_container_width=True):

        input_data = {
            "type": type_,
            "sub_type": sub_type,
            "listing_type": listing_type,
            "tom": float(tom),
            "building_age": building_age,
            "total_floor_count": total_floor_count,
            "floor_no": floor_no,
            "room_count": room_count,
            "size": float(size),
            "heating_type": heating_type,
            "city": city,
            "district": district,
            "neighborhood": neighborhood
        }
        
        try:
            with st.spinner("Анализируем параметры..."):
                response = requests.post("http://localhost:8077/predict", 
                                       json=input_data, 
                                       timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" in result:
                    st.error(f"Ошибка: {result['error']}")
                else:
                    price_rub = result['price_prediction_rub']
                    
                    st.success(f"""
                    ## Предсказанная стоимость:
                    # **{price_rub:,.0f} RUB**
                    """)

                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Цена за м²", 
                                f"{result['calculated_price_per_m2']:,.0f} TRY",
                                f"≈ {result['calculated_price_per_m2'] * 3:,.0f} RUB")
                    
                    with col_b:
                        st.metric("Уверенность", 
                                f"{result['confidence_percentage']}%",
                                f"± {result['confidence_interval_rub']:,.0f} RUB")
                    
                    with col_c:
                        st.metric("Площадь на комнату", 
                                f"{result['size_per_room']:.1f} м²")

                    fig, ax = plt.subplots(figsize=(10, 3))

                    min_price = price_rub - result['confidence_interval_rub']
                    max_price = price_rub + result['confidence_interval_rub']

                    ax.barh(['Цена'], [price_rub], 
                           xerr=[[result['confidence_interval_rub']]], 
                           color='lightgreen', 
                           ecolor='red',
                           capsize=10)

                    ax.scatter([min_price, max_price], [0, 0], 
                             color='red', alpha=0.5, s=100)
                    
                    ax.set_xlabel("Цена (RUB)")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)

                    st.subheader("📈 Влияние параметров на цену")
                    
                    influence_data = {
                        "Параметр": ["Площадь", "Город", "Возраст", "Комнаты", "Район"],
                        "Влияние": ["Высокое", "Очень высокое", "Среднее", "Высокое", "Высокое"],
                        "Примерный эффект": [
                            f"+{(size-65)*500*3:,.0f} RUB",
                            f"+{'Высокий' if city=='İstanbul' else 'Средний'}",
                            f"{'-' if building_age in ['20-30', '30+'] else '+'}",
                            f"+{(room_count-2)*200000*3:,.0f} RUB",
                            "Зависит от престижности"
                        ]
                    }
                    
                    st.table(pd.DataFrame(influence_data))
                    
            else:
                st.error(f"Ошибка сервера: {response.status_code}")
                
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    else:
        st.info("""
        **Заполните параметры слева и нажмите кнопку "Рассчитать"**
        """)

st.markdown("---")
st.markdown("""
**Система оценки недвижимости**  
*Цены в рублях, рассчитаны на основе данных в турецких лирах*
""")