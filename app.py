import streamlit as st
import pandas as pd
import itertools
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (風速・波高 強烈反映版)")

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
            st.error(f"ファイル '{filename}' の読み込みに失敗しました: {e}")
            
    return models[0], models[1], models[2]

# --- 2. 推論＆風速・波高の超強力補正計算 ---
def calculate_win_probability(wind_speed, wave_height, jcd, rno):
    m1, m2, m3 = load_ai_models()
    
    if m1 is None or m2 is None or m3 is None:
        return pd.DataFrame(), "AIモデルのロードに失敗しています。.pklファイルを確認してください。"
    
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

    # ★ 【強烈な風速・波高補正ロジック】
    # スライダーの数値が上がった際、1号艇の確率をダイレクトに叩き落とし、外枠を跳ね上げる
    if wind_speed > 0 or wave_height > 0:
        # 1号艇（イン）への激しいペナルティ（風速・波高に比例して確率が激減）
        penalty_rate = max(0.2, 1.0 - (wind_speed * 0.12) - (wave_height * 0.10))
        df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= penalty_rate
        
        # 4〜6号艇（外枠）への強力なボーナス
        bonus_rate = 1.0 + (wind_speed * 0.15) + (wave_height * 0.12)
        df_features.loc[df_features['racer_boat_number'] >= 4, 'prob_1st'] *= bonus_rate

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

# --- 3. UIと実行制御 ---
col1, col2 = st.columns(2)
with col1: jcd = st.selectbox("開催場コード (01〜24)", [f"{i:02d}" for i in range(1, 25)])
with col2: rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])

manual_wind = st.slider("想定風速 (m)", min_value=0, max_value=10, value=2, step=1)
manual_wave = st.slider("想定波高 (cm)", min_value=0, max_value=15, value=2, step=1)

if st.button("勝率算出＆原稿生成を実行", type="primary"):
    today_display = datetime.now().strftime('%Y年%m月%d日')
    
    with st.spinner("AI推論中..."):
        df_results, error_msg = calculate_win_probability(manual_wind, manual_wave, jcd, rno)
        
        if error_msg:
            st.error(error_msg)
        else:
            st.success(f"推論完了（適用風速: {manual_wind}m / 波高: {manual_wave}cm ※強烈反映版）")
            st.dataframe(df_results, use_container_width=True)
            
            x_text = f"風速{manual_wind}m・波高{manual_wave}cmの荒れ水面補正を最大適用。\n\n本日、場コード{jcd}の{rno}Rにおいて、AIが波乱の特注買い目を検知しました。\n特注買い目はこちら👇\n[noteURL]\n#競艇予想 #ボートレース"
            st.subheader("📱 X集客用テキスト")
            st.code(x_text, language="text")
            
            note_text = f"【{today_display}】AI勝率上位・特注レース(場:{jcd} {rno}R / 風速:{manual_wind}m 波高:{manual_wave}cm)\n\n■無料エリア：\n荒れた水面はインの信頼度を大きく揺るがします。\n当AIは風速・波高による激変データを解析し、高配当を狙う黄金の目のみを公開します。\n\n===== 有料エリア =====\n\n■AI算出 トップ買い目（勝率上位）\n"
            for _, row in df_results.iterrows():
                note_text += f"推奨: 【 {row['買い目']} 】 (AI算出勝率 {row['AI勝率(%)']} %)\n"
            note_text += "\n※投資は自己責任でお願いします。"
            st.subheader("📝 note販売用テキスト")
            st.code(note_text, language="text")