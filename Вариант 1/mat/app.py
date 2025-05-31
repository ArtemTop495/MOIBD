import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from datetime import datetime

# Стиль терминала хакера
st.set_page_config(
    page_title="// GHOST_PROTOCOL",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    * {
        font-family: 'Share Tech Mono', monospace !important;
    }
    body {
        background-color: #000 !important;
        color: #0f0 !important;
    }
    .stTextArea textarea {
        background-color: #111 !important;
        color: #0f0 !important;
        border: 1px solid #0f0 !important;
    }
    .stButton>button {
        background: #0f0 !important;
        color: #000 !important;
        border: none;
        font-weight: bold;
    }
    .st-bb { background-color: transparent !important; }
    .st-at { background-color: #0f0 !important; }
    .terminal {
        border: 1px solid #0f0;
        padding: 15px;
        margin: 10px 0;
        background: #000;
        border-radius: 5px;
    }
    .scan-header {
        color: #0f0;
        text-shadow: 0 0 5px #0f0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data(show_spinner=False)
def _steal_data():
    try:
        df = pd.read_csv('lala.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['text_length'] = df['text'].str.len()
        return df
    except:
        return pd.DataFrame()


data = _steal_data()


tab1, tab2 = st.tabs(["// TEXT_SCAN", "// DATA_MATRIX"])

with tab1:
    st.markdown("<h1 class='scan-header'>TEXT ANALYSIS MODE</h1>", unsafe_allow_html=True)

    input_text = st.text_area(
        "INPUT TEXT:",
        placeholder="[ENTER TEXT TO ANALYZE]",
        height=150
    )

    if st.button("EXECUTE SCAN"):
        if not input_text.strip():
            st.warning("[ERROR] NO INPUT DETECTED")
        else:
            with st.spinner("[SCANNING...]"):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={"text": input_text},
                        timeout=5
                    )

                    if response.status_code == 200:
                        res = response.json()

                        st.markdown(f"""
                        <div class='terminal'>
                        > STATUS: {res["sentiment"].upper()}<br>
                        > CONFIDENCE: {res["confidence"]:.1%}<br>
                        > TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </div>
                        """, unsafe_allow_html=True)

                        # График в стиле терминала
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.bar(["NEG", "POS"],
                               [res["probabilities"]["negative"], res["probabilities"]["positive"]],
                               color=["#f00", "#0f0"])
                        ax.set_facecolor("#000")
                        fig.patch.set_facecolor("#000")
                        ax.tick_params(colors="#0f0")
                        for spine in ax.spines.values():
                            spine.set_color("#0f0")
                        st.pyplot(fig)

                    else:
                        st.error(f"[SERVER ERROR] CODE: {response.status_code}")

                except requests.exceptions.RequestException:
                    st.error("""
                    <div class='terminal'>
                    > CONNECTION FAILED<br>
                    > API SERVER OFFLINE<br>
                    > CHECK SERVER STATUS
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h1 class='scan-header'>DATA MATRIX ACCESS</h1>", unsafe_allow_html=True)

    if data.empty:
        st.warning("[ERROR] DATA STREAM NOT FOUND")
    else:

        cols = st.columns(4)
        metrics = [
            ("TOTAL RECORDS", len(data)),
            ("UNIQUE AGENTS", data['name'].nunique()),
            ("AVG LENGTH", f"{data['text_length'].mean():.1f}"),
            ("POSITIVITY", f"{(data['type'] == 1).mean() * 100:.1f}%")
        ]

        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class='terminal' style='padding:10px;text-align:center;'>
                {label}<br>
                <span style='font-size:1.5em;'>{value}</span>
                </div>
                """, unsafe_allow_html=True)


        st.markdown("#### TEXT LENGTH DISTRIBUTION")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(data['text_length'], bins=30, color="#0f0", edgecolor="#000")
        ax.set_facecolor("#000")
        fig.patch.set_facecolor("#000")
        ax.tick_params(colors="#0f0")
        for spine in ax.spines.values():
            spine.set_color("#0f0")
        st.pyplot(fig)


        st.markdown("#### LEXICAL FREQUENCY MATRIX")
        wc = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='viridis',
            contour_width=1,
            contour_color='#0f0'
        ).generate(" ".join(data['text'].dropna()))

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        fig2.patch.set_facecolor('#000')
        st.pyplot(fig2)


with st.sidebar:
    st.markdown("""
    <div class='terminal'>
    > SYSTEM OVERVIEW<br><br>
    > ACTIVE RECORDS: {records:,}<br>
    > DATA RANGE: {date_min} - {date_max}<br>
    > VERSION: GHOST/2.4.1<br>
    > STATUS: ONLINE
    </div>
    """.format(
        records=len(data) if not data.empty else 0,
        date_min=data['date'].min().date() if not data.empty else "N/A",
        date_max=data['date'].max().date() if not data.empty else "N/A"
    ), unsafe_allow_html=True)

    st.markdown("""
    <div class='terminal' style='margin-top:20px;'>
    > NAVIGATION<br><br>
    // TEXT_SCAN<br>
    - Real-time analysis<br><br>
    // DATA_MATRIX<br>
    - Full dataset stats
    </div>
    """, unsafe_allow_html=True)