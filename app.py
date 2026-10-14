import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import itertools
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (ハイブリッド直前情報対応版)")

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

# --- 2. 出走表（勝率・モーター・F持ち）の自動スクレイピング ---
def fetch_race_card(jcd, rno):
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST).strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={today}"
    
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        racers = []
        tbodies = soup.find_all('tbody', class_='is-fs12')
        
        if not tbodies or len(tbodies) < 6:
            return None, "出走表データが取得できません。"
            
        for i in range(6):
            tbody = tbodies[i]
            try:
                name_div = tbody.find('div', class_='is-fs18')
                name = name_div.find('a').text.strip() if name_div and name_div.find('a') else f"{i+1}号艇選手"

                tds = tbody.find_all('td')
                
                # ★追加：F持ち（フライング）情報の取得（通常 tds[3] に記載）
                try:
                    f_text = tds[3].text.strip()
                    has_f = ("F1" in f_text) or ("F2" in f_text) or ("F3" in f_text)
                except:
                    has_f = False

                # 全国勝率の取得
                try:
                    national_data = list(tds[4].stripped_strings)
                    win_rate = float(national_data[0])
                except:
                    win_rate = 5.0
                    
                # モーター2連対率の取得
                try:
                    motor_data = list(tds[6].stripped_strings)
                    motor_rate = float(motor_data[1])
                except:
                    motor_rate = 30.0
                    
            except Exception as e:
                name = f"{i+1}号艇(エラー)"
                win_rate = 5.0
                motor_rate = 30.0
                has_f = False
                
            racers.append({
                'boat': i + 1,
                'name': name,
                'win_rate': win_rate,
                'motor_rate': motor_rate,
                'has_f': has_f
            })
            
        return pd.DataFrame(racers), None
    except Exception as e:
        return None, f"データ取得エラー ({e})"

# --- 2.5 直前情報（展示タイム）の自動スクレイピング ---
def fetch_beforeinfo(jcd, rno):
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST).strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={today}"
    
    times = {}
    try:
        response = requests.get(url, timeout=5)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        tbodies = soup.find_all('tbody', class_='is-fs12')
        if not tbodies or len(tbodies) < 6:
            return None # まだ展示データが公開されていない
            
        for i in range(6):
            tbody = tbodies[i]
            tds = tbody.find_all('td')
            # 展示タイムは通常 tds[4] に記載
            try:
                ext_time_text = tds[4].text.strip()
                if ext_time_text:
                    times[i+1] = float(ext_time_text)
            except:
                pass
                
        if len(times) == 6:
            return times
        return None
    except:
        return None

# --- 3. 推論＆環境・選手・直前情報統合ロジック ---
def calculate_win_probability(df_racers, dict_exhibition, wind_direction, wind_speed, wave_height, jcd, rno):
    m1, m2, m3 = load_ai_models()
    
    if m1 is None or m2 is None or m3 is None:
        return pd.DataFrame(), "AIモデルのロードに失敗しました。"
    
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
        return pd.DataFrame(), f"推論処理エラー: {e}"

    # ★ 【勝率・F持ち・展示の極大化ブースト】
    boat1_win_rate = df_racers[df_racers['boat'] == 1].iloc[0]['win_rate']
    
    for boat in range(1, 7):
        racer_data = df_racers[df_racers['boat'] == boat].iloc[0]
        w_rate = racer_data['win_rate']
        m_rate = racer_data['motor_rate']
        
        # 1. 基本能力値の算出
        ability_multiplier = 1.0 + ((w_rate - 5.0) * 0.20) + ((m_rate - 30.0) * 0.02)
        
        # 2. 強者プレミアムとF持ちペナルティ（致命的減点）
        if w_rate >= 6.5:
            ability_multiplier *= 1.30 
        elif w_rate < 4.5:
            ability_multiplier *= 0.60 
            
        if racer_data['has_f']:
            # F持ちはスタートで遅れるため勝率を強制的に下げる
            ability_multiplier *= 0.60

        df_features.loc[df_features['racer_boat_number'] == boat, 'prob_1st'] *= max(0.05, ability_multiplier)

    # 3. 【直前情報：展示タイム1位の爆発的加点】
    if dict_exhibition:
        # 展示タイムが一番小さい（速い）艇の号艇番号を取得
        fastest_boat = min(dict_exhibition, key=dict_exhibition.get)
        # その艇の勝率・連対率を跳ね上げる（隠れ超抜機の検知）
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_1st'] *= 1.40
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_2nd'] *= 1.30
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_3rd'] *= 1.30

    # 4. 【1号艇の鉄板・崩壊ロジック】
    if boat1_win_rate >= 6.5 and not df_racers[df_racers['boat'] == 1].iloc[0]['has_f']:
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.50
    elif boat1_win_rate < 5.0 or df_racers[df_racers['boat'] == 1].iloc[0]['has_f']:
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.40
        df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.40

    # 5. 【環境要因のハード補正】
    if wind_direction == "向かい風" and (wind_speed >= 4 or wave_height >= 5):
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.10 
        df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.50 
    elif wind_direction == "追い風" and wave_height < 3:
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.30 

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
    
    # UI表示用に展示タイム最速艇の情報を返す
    fastest_msg = ""
    if dict_exhibition:
        fastest_boat = min(dict_exhibition, key=dict_exhibition.get)
        fastest_time = dict_exhibition[fastest_boat]
        fastest_msg = f"🔥 展示タイム最速検知: {fastest_boat}号艇 ({fastest_time}秒) -> 確率ブースト適用済"
        
    return df_results, fastest_msg, None

# --- 4. UIと実行制御 ---
col1, col2 = st.columns(2)
with col1: jcd = st.selectbox("開催場コード (01〜24)", [f"{i:02d}" for i in range(1, 25)])
with col2: rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])

wind_direction = st.selectbox("風向", ["追い風", "向かい風", "横風・その他"])
manual_wind = st.slider("想定風速 (m)", min_value=0, max_value=10, value=2, step=1)
manual_wave = st.slider("想定波高 (cm)", min_value=0, max_value=15, value=2, step=1)

if st.button("AI推論実行 (出走表・直前情報自動取得)", type="primary"):
    JST = timezone(timedelta(hours=+9), 'JST')
    today_display = datetime.now(JST).strftime('%Y年%m月%d日')
    
    with st.spinner("データスクレイピング中..."):
        df_racers, fetch_error = fetch_race_card(jcd, rno)
        dict_exhibition = fetch_beforeinfo(jcd, rno)
        
    if fetch_error:
        st.error(fetch_error)
    else:
        # UIでの状況表示
        if dict_exhibition:
            st.success("✅ 【直前予想モード】出走表および展示タイムの取得に成功しました。")
        else:
            st.info("ℹ️ 【事前予想モード】展示タイムは未公開です。出走表（勝率・モーター・F持ち）のみで推論します。")
            
        st.write("▼ 取得した出走表データ（True=F持ちペナルティ対象）")
        st.dataframe(df_racers, use_container_width=True)
        
        with st.spinner("AI推論中..."):
            df_results, fastest_msg, error_msg = calculate_win_probability(df_racers, dict_exhibition, wind_direction, manual_wind, manual_wave, jcd, rno)
            
            if error_msg:
                st.error(error_msg)
            else:
                if fastest_msg:
                    st.warning(fastest_msg)
                    
                st.dataframe(df_results, use_container_width=True)
                
                mode_str = "【直前情報加味】" if dict_exhibition else "【事前予想】"
                x_text = f"🔥 {mode_str} AI推論完了\n\n本日、場コード{jcd}の{rno}Rにおいて、極秘の特注買い目を検知しました。\n特注買い目はこちら👇\n[noteURL]\n#競艇予想 #ボートレース"
                st.subheader("📱 X集客用テキスト")
                st.code(x_text, language="text")
                
                note_text = f"【{today_display}】{mode_str} 特注レース(場:{jcd} {rno}R)\n\n■無料エリア：\n当AIは選手の勝率、モーター、F持ち状態{ '、直前の展示タイム' if dict_exhibition else '' }をリアルタイムで取得し、統合しています。\n\n===== 有料エリア =====\n\n■AI算出 トップ買い目\n"
                for _, row in df_results.iterrows():
                    note_text += f"推奨: 【 {row['買い目']} 】 (AI算出勝率 {row['AI勝率(%)']} %)\n"
                note_text += "\n※投資は自己責任でお願いします。"
                st.subheader("📝 note販売用テキスト")
                st.code(note_text, language="text")