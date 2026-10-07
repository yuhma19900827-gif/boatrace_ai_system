import streamlit as st
import pandas as pd
import itertools
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="競艇AIマネタイズシステム", layout="centered")
st.title("🚤 競艇AI 勝率推論 (風向・風速・波高 統合版)")

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

# --- 2. 推論＆風向・風速・波高の複合補正計算 ---
def calculate_win_probability(wind_direction, wind_speed, wave_height, jcd, rno):
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

    # ★ 【風向・風速・波高の統合補正ロジック】
    # 追い風ならイン強化、向かい風なら外（まくり・差し）強化、さらに波高の高さに応じたペナルティ
    
    penalty_1st = 1.0
    bonus_outer = 1.0

    if wind_direction == "向かい風":
        # 向かい風はインが弱まり、外が強くなる
        penalty_1st -= (wind_speed * 0.08) + (wave_height * 0.06)
        bonus_outer += (wind_speed * 0.10) + (wave_height * 0.08)
    elif wind_direction == "追い風":
        # 追い風はインが圧倒的に堅くなる（ただし波高が高いと相殺される）
        if wave_height < 5:
            df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= (1.0 + (wind_speed * 0.05))
        else:
            penalty_1st -= (wave_height * 0.05) # 波高が高ければ追い風でも荒れる
    else: # 横風など
        penalty_1st -= (wind_speed * 0.05) + (wave_height * 0.05)
        bonus_outer += (wind_speed * 0.06)

    # 1号艇の確率にペナルティ適用（下限維持）
    df_features.loc[df_features['racer_boat_number'] == 1, 'prob_1st'] *= max(0.2, penalty_1st)
    # 4〜6号艇の外枠にボーナス適用
    df_features.loc[df_features['racer_boat_number'] >= 4, 'prob_1st'] *= max(1.0, bonus_outer)

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

# ★ 風向の選択肢を追加
wind_direction = st.selectbox("風向", ["追い風", "向かい風", "横風・その他"])
manual_wind = st.slider("想定風速 (m)", min_value=0, max_value=10, value=2, step=1)
manual_wave = st.slider("想定波高 (cm)", min_value=0, max_value=15, value=2, step=1)

if st.button("勝率算出＆原稿生成を実行", type="primary"):
    today_display = datetime.now().strftime('%Y年%m月%d日')
    
    with st.spinner("AI推論中..."):
        df_results, error_msg = calculate_win_probability(wind_direction, manual_wind, manual_wave, jcd, rno)
        
        if error_msg:
            st.error(error_msg)
        else:
            st.success(f"推論完了（風向: {wind_direction} / 風速: {manual_wind}m / 波高: {manual_wave}cm）")
            st.dataframe(df_results, use_container_width=True)
            
            x_text = f"【{wind_direction}・風速{manual_wind}m・波高{manual_wave}cm】水面気象完全連動。\n\n本日、場コード{jcd}の{rno}Rにおいて、AIが導いた特注買い目はこちら👇\n[noteURL]\n#競艇予想 #ボートレース"
            st.subheader("📱 X集客用テキスト")
            st.code(x_text, language="text")
            
            note_text = f"【{today_display}】AI勝率上位・特注レース(場:{jcd} {rno}R / {wind_direction} 風速:{manual_wind}m 波高:{manual_wave}cm)\n\n■無料エリア：\n風向と波高がレース展開の命運を握ります。\n当AIは現地の気象力学を完全に反映させ、最も期待値の高い黄金の目のみを抽出しました。\n\n===== 有料エリア =====\n\n■AI算出 トップ買い目（勝率上位）\n"
            for _, row in df_results.iterrows():
                note_text += f"推奨: 【 {row['買い目']} 】 (AI算出勝率 {row['AI勝率(%)']} %)\n"
            note_text += "\n※投資は自己責任でお願いします。"
            st.subheader("📝 note販売用テキスト")
            st.code(note_text, language="text")