import streamlit as st
import requests
import os
# Обнуление статусов прокси для корректного подключения
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

st.title("Предсказание кластера статьи")
# Поле ввода
input_text = st.text_area ("Введите описание статьи", height=208)
if st.button("Предсказать"):
    if input_text == "":
        st.write(f"Введите текст")
    else:
        data = {
            "text": input_text
        }
        url="http://127.0.0.1:8000/predict"
        response = requests.post(url, json=data)
        result = response.json()
        clust = result.get("cluster")
        # Вывод предсказанного кластера
        st.markdown(f"""
        #### Предсказанный кластер
        """)
        st.write(f"Кластер: {clust[0]}")
        st.write(f"{clust[1]}")