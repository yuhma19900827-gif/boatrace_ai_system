import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import itertools
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (スクレイピング完全対応版)")

# --- 1. AIモデルの安全読み込み ---
@st.cache_resource
def load_ai_models():
    models = []
    filenames = ['lgbm_model_1st (2).pkl', 'lgbm_model_2nd.pkl', 'lgbm_model_3rd.pkl']
    
    for filename in filenames:
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
        except Exception as e:
            models.append(None)
            
    return models[0], models[1], models[2]

# --- 2. 出走表（選手データ）の全自動スクレイピング ---
def fetch_race_card(jcd, rno):
    today = datetime.now().strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={today}"
    
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        racers = []
        tbodies = soup.find_all('tbody', class_='is-fs12')
        
        if not tbodies or len(tbodies) < 6:
            return None, "出走表データが取得できません。開催日か場コードを確認してください。"
            
        for i in range(6):
            tbody = tbodies[i]
            try:
                # 選手名の抽出
                name_div = tbody.find('div', class_='is-fs18')
                name = name_div.find('a').text.strip() if name_div and name_div.find('a') else f"{i+1}号艇選手"

                # 各セルの取得
                tds = tbody.find_all('td')
                
                # ★修正ポイント：セル内の文字列を分割して正確に抽出する
                # tds[4] = 全国成績（1行目: 勝率, 2行目: 2連対率, 3行目: 3連対率）
                try:
                    national_data = list(tds[4].stripped_strings)
                    win_rate = float(national_data[0]) # 1つ目の数値（勝率）を取得
                except:
                    win_rate = 5.0
                    
                # tds[6] = モーター成績（1行目: No, 2行目: 2連対率, 3行目: 3連対率）
                try:
                    motor_data = list(tds[6].stripped_strings)
                    motor_rate = float(motor_data[1]) # 2つ目の数値（2連対率）を取得
                except:
                    motor_rate = 30.0
                    
            except Exception as e:
                name = f"{i+1}号艇(パース例外)"
                win_rate = 5.0
                motor_rate = 30.0
                
            racers.append({
                'boat': i + 1,
                'name': name,
                'win_rate': win_rate,
                'motor_rate': motor_rate
            })
            
        return pd.DataFrame(racers), None
    except Exception as e:
        return None, f"データ取得エラー: 通信に失敗しました ({e})"

# --- 3. 推論＆環境・選手統合ロジック ---
def calculate_win_probability(df_racers, wind_direction, wind_speed, wave_height, jcd, rno):
    m1, m2, m3 = load_ai_models()
    
    if m1 is None or m2 is None or m3 is None:
        return pd.DataFrame(), "AIモデルのロードに失敗しました。.pklファイルを確認してください。"
    
    test_features = []
    for boat in range(1, 7):
        test_features.append({
            'race_stadium_number': int(jcd),
            'race_number': int(rno),
            'racer_boat_number': boat,
            'racer_course_number': boat, 
            'race_wind': int(wind_speed)
        })
    df_features = pd.DataFrame(test_features)
    
    try:
        df_features['prob_1st'] = m1.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_2nd'] = m2.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_3rd'] = m3.predict(df_features, predict_disable_shape_check=True)
    except Exception as e:
        return pd.DataFrame(), f"推論処理でエラーが発生しました: {e}"

    # ★ 【選手データ ＋ 風・波の完全統合オーバーライド】
    for boat in range(1, 7):
        racer_data = df_racers[df_racers['boat'] == boat].iloc[0]
        
        # 勝率とモーター率による補正倍率の計算
        ability_multiplier = 1.0 + ((racer_data['win_rate'] - 5.0) * 0.1) + ((racer_data['motor_rate'] - 30.0) * 0.01)
        df_features.loc[df_features['racer_boat_number'] == boat, 'prob_1st'] *= max(0.1, ability_multiplier)

    # 環境要因のハード補正
    if wind_direction == "向かい風" and (wind_speed >= 4 or wave_height >= 5):
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.15 
        df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.40 
    elif wind_direction == "追い風" and wave_height < 3:
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.50 

    results = []
    for combo in itertools.permutations([1, 2, 3, 4, 5, 6], 3):
        b1, b2, b3 = combo
        p1 = df_features[df_features['racer_boat_number'] == b1]['prob_1st'].values[0]
        p2 = df_features[df_features['racer_boat_number'] == b2]['prob_2nd'].values[0]
        p3 = df_features[df_features['racer_boat_number'] == b3]['prob_3rd'].values[0]
        
        combined_prob = p1 * p2 * p3
        results.append({
            '買い目': f"{b1}-{b2}-{b3}",
            'AI勝率(%)': round(combined_prob * 100, 2)
        })
                
    df_results = pd.DataFrame(results).sort_values('AI勝率(%)', ascending=False).head(10)
    return df_results, None

# --- 4. UIと実行制御 ---
col1, col2 = st.columns(2)
with col1: jcd = st.selectbox("開催場コード (01〜24)", [f"{i:02d}" for i in range(1, 25)])
with col2: rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])

wind_direction = st.selectbox("風向", ["追い風", "向かい風", "横風・その他"])
manual_wind = st.slider("想定風速