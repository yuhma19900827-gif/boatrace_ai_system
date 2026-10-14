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
    
    max_retries = 3
    soup = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            break
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"通信エラー ({e})"
            time.sleep(2)

    if not soup:
        return None, "HTML解析エラー"

    try:
        racers = []
        # ★究極修正：「toban=」が含まれる選手プロフィールへのリンクを直接狙撃する
        racer_links = soup.find_all('a', href=lambda href: href and 'toban=' in href)
        
        for a_tag in racer_links:
            # リンクの親要素（tbody）を特定し、選手のデータブロックを完全隔離
            tbody = a_tag.find_parent('tbody')
            if not tbody:
                continue
                
            # ブロック内から1〜6号艇のカラータグを探して枠番を確定
            boat = None
            for i in range(1, 7):
                if tbody.find('td', class_=f'is-boatColor{i}'):
                    boat = i
                    break
                    
            if not boat:
                continue
                
            # すでにこの号艇がリストにいる場合はスキップ（重複取得防止）
            if any(r['boat'] == boat for r in racers):
                continue
                
            name = a_tag.text.strip()
            tds = tbody.find_all('td')
            
            # F持ち情報の取得
            has_f = False
            for td in tds:
                txt = td.text
                if "F1" in txt or "F2" in txt or "F3" in txt:
                    has_f = True
                    break

            try:
                national_data = list(tds[4].stripped_strings)
                win_rate = float(national_data[0])
            except:
                win_rate = 5.0
                
            try:
                motor_data = list(tds[6].stripped_strings)
                motor_rate = float(motor_data[1])
            except:
                motor_rate = 30.0
                
            racers.append({
                'boat': boat,
                'name': name,
                'win_rate': win_rate,
                'motor_rate': motor_rate,
                'has_f': has_f
            })
            
        # 枠番順に並び替え
        racers = sorted(racers, key=lambda x: x['boat'])
        
        if not racers or len(racers) < 2:
            return None, "出走表がまだ公開されていないか、データが取得できません。"
            
        return pd.DataFrame(racers), None
    except Exception as e:
        return None, f"データ抽出エラー ({e})"

# --- 2.5 直前情報（展示タイム）の自動スクレイピング ---
def fetch_beforeinfo(jcd, rno):
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST).strftime('%Y%m%d')
    url = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={today}"
    
    max_retries = 3
    soup = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=20)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            break
        except:
            if attempt == max_retries - 1:
                return None
            time.sleep(2)

    if not soup:
        return None

    times = {}
    try:
        # 展示情報ページでも「toban=」のリンクを基準にブロックを特定
        racer_links = soup.find_all('a', href=lambda href: href and 'toban=' in href)
        
        for a_tag in racer_links:
            tbody = a_tag.find_parent('tbody')
            if not tbody:
                continue
                
            boat = None
            for i in range(1, 7):
                if tbody.find('td', class_=f'is-boatColor{i}'):
                    boat = i
                    break
                    
            if not boat or boat in times:
                continue
                
            # その選手のブロック内から「6.x〜7.x秒」の展示タイムを抽出
            tds = tbody.find_all('td')
            for td in tds:
                txt = td.text.strip()
                try:
                    val = float(txt)
                    if 6.0 <= val <= 7.5:
                        times[boat] = val
                        break
                except:
                    continue
                        
        if len(times) > 0:
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
        # 欠場などでデータがない場合はスキップしないようダミーを入れる
        if boat in df_racers['boat'].values:
            test_features.append({
                'race_stadium_number': int(jcd),
                'race_number': int(rno),
                'racer_boat_number': boat,
                'racer_course_number': boat, 
                'race_wind': int(wind_speed)
            })
    
    if len(test_features) != 6:
        return pd.DataFrame(), "6艇分のデータが揃っていません（欠場艇あり）。"
        
    df_features = pd.DataFrame(test_features)
    
    try:
        df_features['prob_1st'] = m1.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_2nd'] = m2.predict(df_features, predict_disable_shape_check=True)
        df_features['prob_3rd'] = m3.predict(df_features, predict_disable_shape_check=True)
    except Exception as e:
        return pd.DataFrame(), f"推論処理エラー: {e}"

    boat1_win_rate = df_racers[df_racers['boat'] == 1].iloc[0]['win_rate'] if 1 in df_racers['boat'].values else 5.0
    
    for boat in range(1, 7):
        racer_data = df_racers[df_racers['boat'] == boat].iloc[0]
        w_rate = racer_data['win_rate']
        m_rate = racer_data['motor_rate']
        
        ability_multiplier = 1.0 + ((w_rate - 5.0) * 0.20) + ((m_rate - 30.0) * 0.02)
        
        if w_rate >= 6.5:
            ability_multiplier *= 1.30 
        elif w_rate < 4.5:
            ability_multiplier *= 0.60 
            
        if racer_data['has_f']:
            ability_multiplier *= 0.60

        df_features.loc[df_features['racer_boat_number'] == boat, 'prob_1st'] *= max(0.05, ability_multiplier)

    if dict_exhibition:
        fastest_boat = min(dict_exhibition, key=dict_exhibition.get)
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_1st'] *= 1.40
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_2nd'] *= 1.30
        df_features.loc[df_features['racer_boat_number'] == fastest_boat, 'prob_3rd'] *= 1.30

    # 1号艇のデータが存在する場合のみ鉄板・崩壊ロジックを適用
    boat1_data = df_racers[df_racers['boat'] == 1]
    if not boat1_data.empty:
        has_f_1 = boat1_data.iloc[0]['has_f']
        if boat1_win_rate >= 6.5 and not has_f_1:
            df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 1.50
        elif boat1_win_rate < 5.0 or has_f_1:
            df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= 0.40
            df_features.loc[df_features['racer_boat_number'] >= 3, 'prob_1st'] *= 1.40

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
    
    fastest_msg = ""
    if dict_exhibition:
        fastest_boat = min(dict_exhibition, key=dict_exhibition.get)
        fastest_time = dict_exhibition[fastest_boat]
        fastest_msg = f"🔥 展示タイム最速検知: {fastest_boat}号艇 ({fastest_time}秒) -> AI確率ブースト適用済"
        
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
    
    with st.spinner("データスクレイピング中（通信待機中...）"):
        df_racers, fetch_error = fetch_race_card(jcd, rno)
        dict_exhibition = None
        if not fetch_error:
            # df_racers を渡さずとも独立して取得可能に変更済
            dict_exhibition = fetch_beforeinfo(jcd, rno)
        
    if fetch_error:
        st.error(fetch_error)
    else:
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