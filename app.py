import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import itertools
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (ハイブリッド直前情報対応版)")

@st.cache_resource
def load_ai_models():
    models = []
    filenames = ['lgbm_model_1st (2).pkl', 'lgbm_model_2nd.pkl', 'lgbm_model_3rd.pkl']
    for filename in filenames:
        try:
            with open(filename, 'rb') as f:
                models.append(pickle.load(f))
        except Exception:
            models.append(None)
    return models[0], models[1], models[2]

def fetch_race_card(jcd, rno):
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST).strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={today}"
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            break
        except:
            if attempt == 2: return None, "通信エラー: 公式サイト混雑"
            time.sleep(2)

    try:
        racers = []
        for i in range(1, 7):
            # 確実に対象号艇のカラーセルを狙撃
            boat_td = soup.find(class_=f'is-boatColor{i}')
            if not boat_td: continue
            
            # 検索範囲を「その艇の行（tr）」だけに完全限定する（他の選手が混ざるのを物理的に防止）
            tr = boat_td.find_parent('tr')
            if not tr: continue

            try:
                name_div = tr.find('div', class_='is-fs18')
                name = name_div.find('a').text.strip() if name_div and name_div.find('a') else (name_div.text.strip() if name_div else f"{i}号艇")
                tds = tr.find_all('td')
                
                has_f = False
                if len(tds) > 3:
                    has_f = any(f_val in tds[3].text for f_val in ["F1", "F2", "F3"])
                
                win_rate = 5.0
                if len(tds) > 4:
                    try: win_rate = float(list(tds[4].stripped_strings)[0])
                    except: pass
                    
                motor_rate = 30.0
                if len(tds) > 6:
                    try: motor_rate = float(list(tds[6].stripped_strings)[1])
                    except: pass
            except:
                name, win_rate, motor_rate, has_f = f"{i}号艇", 5.0, 30.0, False
                
            racers.append({'boat': i, 'name': name, 'win_rate': win_rate, 'motor_rate': motor_rate, 'has_f': has_f})
            
        if len(racers) < 2: return None, "出走表が未公開か取得不可"
        return pd.DataFrame(racers), None
    except Exception as e:
        return None, f"データ抽出エラー: {e}"

def fetch_beforeinfo(jcd, rno):
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST).strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={today}"
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            break
        except:
            if attempt == 2: return None
            time.sleep(2)

    times = {}
    try:
        for i in range(1, 7):
            boat_td = soup.find(class_=f'is-boatColor{i}')
            if not boat_td: continue
            
            tr = boat_td.find_parent('tr')
            if not tr: continue
            
            for td in tr.find_all('td'):
                try:
                    val = float(td.text.strip())
                    if 6.0 <= val <= 7.5:
                        times[i] = val
                        break
                except: continue
        return times if times else None
    except: return None

def calculate_win_probability(df_racers, dict_exhibition, wind_direction, wind_speed, wave_height, jcd, rno):
    m1, m2, m3 = load_ai_models()
    if not all([m1, m2, m3]): 
        return pd.DataFrame(), "", "AIモデル読込エラー。"
    
    test_features = [{'race_stadium_number': int(jcd), 'race_number': int(rno), 'racer_boat_number': b, 'racer_course_number': b, 'race_wind': int(wind_speed)} for b in range(1, 7) if b in df_racers['boat'].values]
    if len(test_features) != 6: 
        return pd.DataFrame(), "", "6艇分のデータ不足（欠場等）。"
        
    df_features = pd.DataFrame(test_features)
    
    try:
        df_features['prob_1st'] = m1.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_2nd'] = m2.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_3rd'] = m3.predict(df_features, predict_disable_shape_check=True)
    except Exception as e:
        return pd.DataFrame(), "", f"AI推論エラー: {e}"

    boat1_df = df_racers[df_racers['boat'] == 1]
    boat1_win_rate = boat1_df.iloc[0]['win_rate'] if not boat1_df.empty else 5.0
    
    for boat in range(1, 7):
        racer_df = df_racers[df_racers['boat'] == boat]
        if racer_df.empty: continue
        racer_data = racer_df.iloc[0]
        
        am = 1.0 + ((racer_data['win_rate'] - 5.0) * 0.20) + ((racer_data['motor_rate'] - 30.0) * 0.02)
        if racer_data['win_rate'] >= 6.5: am *= 1.30 
        elif racer_data['win_rate'] < 4.5: am *= 0.60 
        if racer_data['has_f']: am *= 0.60
        df_features.loc[df_features['racer_boat_number'] == boat, 'prob_1st'] *= max(0.05, am)

    if dict_exhibition:
        fastest = min(dict_exhibition, key=dict_exhibition.get)
        df_features.loc[df_features['racer_boat_number'] == fastest, ['prob_1st', 'prob_2nd', 'prob_3rd']] *= [1.4, 1.3, 1.3]

    if not boat1_df.empty:
        if boat1_win_rate >= 6.5 and not boat1_df.iloc[0]['has_f']: df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.50
        elif boat1_win_rate < 5.0 or boat1_df.iloc[0]['has_f']: 
            df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.40
            df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.40

    if wind_direction == "向かい風" and (wind_speed >= 4 or wave_height >= 5):
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.10 
        df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.50 
    elif wind_direction == "追い風" and wave_height < 3:
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.30 

    results = []
    for combo in itertools.permutations([1,2,3,4,5,6], 3):
        p1 = df_features[df_features['racer_boat_number'] == combo[0]]['prob_1st'].values[0]
        p2 = df_features[df_features['racer_boat_number'] == combo[1]]['prob_2nd'].values[0]
        p3 = df_features[df_features['racer_boat_number'] == combo[2]]['prob_3rd'].values[0]
        results.append({'買い目': f"{combo[0]}-{combo[1]}-{combo[2]}", 'AI勝率(%)': round(p1 * p2 * p3 * 100, 2)})
                
    df_results = pd.DataFrame(results).sort_values('AI勝率(%)', ascending=False).head(10)
    fastest_msg = f"🔥 展示タイム最速検知: {fastest}号艇 ({dict_exhibition[fastest]}秒) -> ブースト適用" if dict_exhibition else ""
    return df_results, fastest_msg, ""

col1, col2 = st.columns(2)
with col1: jcd = st.selectbox("開催場コード (01〜24)", [f"{i:02d}" for i in range(1, 25)])
with col2: rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])
wind_direction = st.selectbox("風向", ["追い風", "向かい風", "横風・その他"])
manual_wind = st.slider("想定風速 (m)", 0, 10, 2)
manual_wave = st.slider("想定波高 (cm)", 0, 15, 2)

if st.button("AI推論実行 (出走表・直前情報自動取得)", type="primary"):
    today_display = datetime.now(timezone(timedelta(hours=+9), 'JST')).strftime('%Y年%m月%d日')
    with st.spinner("データ取得中..."):
        df_racers, fetch_error = fetch_race_card(jcd, rno)
        dict_exhibition = fetch_beforeinfo(jcd, rno) if not fetch_error else None
        
    if fetch_error:
        st.error(fetch_error)
    else:
        if dict_exhibition: st.success("✅ 【直前予想モード】出走表および展示タイムの取得に成功しました。")
        else: st.info("ℹ️ 【事前予想モード】展示タイム未公開。出走表データのみで計算します。")
            
        st.write("▼ 取得した出走表データ（True=F持ちペナルティ対象）")
        st.dataframe(df_racers, use_container_width=True)
        
        with st.spinner("AI推論中..."):
            res = calculate_win_probability(df_racers, dict_exhibition, wind_direction, manual_wind, manual_wave, jcd, rno)
            df_results = res[0] if len(res) > 0 else pd.DataFrame()
            fastest_msg = res[1] if len(res) > 1 else ""
            error_msg = res[2] if len(res) > 2 else "不明なエラー"
            
            if error_msg:
                st.error(error_msg)
            else:
                if fastest_msg: st.warning(fastest_msg)
                st.dataframe(df_results, use_container_width=True)
                
                mode_str = "【直前情報加味】" if dict_exhibition else "【事前予想】"
                st.subheader("📝 note販売用テキスト")
                note_text = f"【{today_display}】{mode_str} 特注レース(場:{jcd} {rno}R)\n\n■無料エリア：\n当AIは選手のデータ{ 'と直前の展示タイム' if dict_exhibition else '' }を統合しています。\n\n===== 有料エリア =====\n\n"
                for _, row in df_results.iterrows(): note_text += f"推奨: 【 {row['買い目']} 】\n"
                st.code(note_text, language="text")