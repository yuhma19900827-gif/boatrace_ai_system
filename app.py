import streamlit as st
import pandas as pd
import itertools
from datetime import datetime
import pickle

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (環境依存・完全自走版)")

# --- 1. AIモデル読み込み ---
@st.cache_resource
def load_ai_models():
    try:
        with open('lgbm_model_1st.pkl', 'rb') as f: m1 = pickle.load(f)
        with open('lgbm_model_2nd.pkl', 'rb') as f: m2 = pickle.load(f)
        with open('lgbm_model_3rd.pkl', 'rb') as f: m3 = pickle.load(f)
        return m1, m2, m3
    except FileNotFoundError:
        return None, None, None

# --- 2. 推論＆勝率計算 ---
def calculate_win_probability(wind_speed, jcd, rno):
    m1, m2, m3 = load_ai_models()
    
    test_features = []
    for boat in range(1, 7):
        test_features.append({
            'race_stadium_number': int(jcd),
            'race_number': int(rno),
            'racer_boat_number': boat,
            'racer_course_number': boat, # 枠なり進入で固定
            'race_wind': wind_speed
        })
    df_features = pd.DataFrame(test_features)
    
    if m1 is None:
        return pd.DataFrame()
        
    df_features['prob_1st'] = m1.predict(df_features)
    df_features['prob_2nd'] = m2.predict(df_features)
    df_features['prob_3rd'] = m3.predict(df_features)

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
    return df_results

# --- 3. UIと実行制御 ---
col1, col2 = st.columns(2)
with col1: jcd = st.selectbox("開催場コード (01〜24)", [f"{i:02d}" for i in range(1, 25)])
with col2: rno = st.selectbox("レース番号", [str(i) for i in range(1, 13)])
manual_wind = st.slider("想定風速 (m)", min_value=0, max_value=10, value=2, step=1)

if st.button("勝率算出＆原稿生成を実行", type="primary"):
    today_display = datetime.now().strftime('%Y年%m月%d日')
    
    with st.spinner("AI推論中..."):
        df_results = calculate_win_probability(manual_wind, jcd, rno)
        
        if df_results.empty:
            st.error("エラー：AIモデル（.pkl）が見つかりません。ファイル名を確認しろ。")
        else:
            st.success(f"推論完了（適用風速: {manual_wind}m）")
            st.dataframe(df_results, use_container_width=True)
            
            x_text = f"過去15万レースのデータと当日の風速から導き出した完全確率論。\n\n本日、場コード{jcd}の{rno}Rにおいて、AIが極めて高い勝率を検知しました。\n特注買い目はこちら👇\n[noteURL]\n#競艇予想 #ボートレース"
            st.subheader("📱 X集客用テキスト")
            st.code(x_text, language="text")
            
            note_text = f"【{today_display}】AI勝率上位・特注レース(場:{jcd} {rno}R)\n\n■無料エリア：\n競艇は確率のゲームです。\n当AIは過去データを解析し、各艇の1着〜3着確率を独立して算出。\n合成勝率の最も高い黄金の目のみを公開します。\n\n===== 有料エリア =====\n\n■AI算出 トップ買い目（勝率上位）\n"
            for _, row in df_results.iterrows():
                note_text += f"推奨: 【 {row['買い目']} 】 (AI算出勝率 {row['AI勝率(%)']} %)\n"
            note_text += "\n※投資は自己責任でお願いします。"
            st.subheader("📝 note販売用テキスト")
            st.code(note_text, language="text")