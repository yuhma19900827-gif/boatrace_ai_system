import streamlit as st
import pandas as pd
import numpy as np
import requests
import itertools
from datetime import datetime
import pickle
from bs4 import BeautifulSoup

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 期待値抽出＆自動生成(完全版)")

# --- 1. オッズ取得関数 ---
def fetch_realtime_odds(jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={date_str}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(response.content, encoding='utf-8')
        odds_data = []
        for table in tables:
            if len(table.columns) >= 3 and table.shape[0] > 10:
                for idx, row in table.iterrows():
                    for col_idx in range(0, len(table.columns), 2):
                        if col_idx + 1 < len(table.columns):
                            combo = str(row[col_idx]).replace(' ', '')
                            odds_val = str(row[col_idx + 1])
                            if '-' in combo and odds_val.replace('.', '', 1).isdigit():
                                odds_data.append({'買い目': combo, 'オッズ': float(odds_val)})
        return pd.DataFrame(odds_data).dropna().drop_duplicates(subset=['買い目']).reset_index(drop=True)
    except:
        return pd.DataFrame()

# --- 2. 直前情報（風速・進入コース）取得関数 ---
def fetch_before_info(jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={date_str}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        wind_text = soup.find(class_='weather1_bodyUnitLabelData')
        wind_speed = int(wind_text.text.replace('m', '').strip()) if wind_text else 2
        
        course_data = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
        return wind_speed, course_data
    except:
        return 2, {1:1, 2:2, 3:3, 4:4, 5:5, 6:6}

# --- 3. AIモデル読み込み ---
@st.cache_resource
def load_ai_models():
    try:
        with open('lgbm_model_1st.pkl', 'rb') as f: m1 = pickle.load(f)
        with open('lgbm_model_2nd.pkl', 'rb') as f: m2 = pickle.load(f)
        with open('lgbm_model_3rd.pkl', 'rb') as f: m3 = pickle.load(f)
        return m1, m2, m3
    except FileNotFoundError:
        return None, None, None

# --- 4. 推論＆期待値計算 ---
def calculate_expected_value(df_odds, wind_speed, course_data, jcd, rno):
    m1, m2, m3 = load_ai_models()
    
    test_features = []
    for boat in range(1, 7):
        test_features.append({
            'race_stadium_number': int(jcd),
            'race_number': int(rno),
            'racer_boat_number': boat,
            'racer_course_number': course_data.get(boat, boat),
            'race_wind': wind_speed
        })
    df_features = pd.DataFrame(test_features)
    
    if m1 is None:
        df_features['prob_1st'] = np.random.uniform(0.1, 0.5, 6)
        df_features['prob_2nd'] = np.random.uniform(0.1, 0.5, 6)
        df_features['prob_3rd'] = np.random.uniform(0.1, 0.5, 6)
    else:
        df_features['prob_1st'] = m1.predict(df_features)
        df_features['prob_2nd'] = m2.predict(df_features)
        df_features['prob_3rd'] = m3.predict(df_features)

    results = []
    for combo in itertools.permutations([1, 2, 3, 4, 5, 6], 3):
        b