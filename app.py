import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="EPL Smart Predictor", layout="wide")

# --- LOAD MODELS & SCALER ---
@st.cache_resource
def load_assets():
    scaler = joblib.load('models/scaler.pkl')
    models = {
        "KNN": joblib.load('models/knn_model.pkl'),
        "Logistic Regression": joblib.load('models/logistic_model.pkl'),
        "SVM": joblib.load('models/svm_model.pkl'),
        "Naive Bayes": joblib.load('models/nb_model.pkl'),
        "Neural Network": joblib.load('models/nn_model.pkl')
    }
    return scaler, models

scaler, models = load_assets()

# --- SIDEBAR: MODEL & QUESTION SELECTION ---
st.sidebar.header("Control Panel")
selected_model_name = st.sidebar.selectbox("Select AI Brain", list(models.keys()))
prediction_type = st.sidebar.selectbox("Choose Prediction Type", [
    "1. Straight Win/Loss",
    "2. Win Probability %",
    "3. Upset Alert",
    "4. 'Safe Bet' Filter",
    "5. Goal Fest Prediction",
    "6. Defensive Battle",
    "7. Form Check",
    "8. Points Gap Analysis",
    "9. Algorithm Comparison",
    "10. Feature Importance"
])

# --- MAIN INTERFACE: MATCH INPUTS ---
st.title("⚽ EPL Smart Prediction Dashboard")
st.markdown(f"Currently using: **{selected_model_name}**")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Team Strength")
    htp = st.number_input("Home Team Total Points", min_value=0, value=30)
    atp = st.number_input("Away Team Total Points", min_value=0, value=25)
    htgd = st.number_input("Home Goal Difference", value=5)
    atgd = st.number_input("Away Goal Difference", value=-2)

with col2:
    st.subheader("Current Form (Last 5)")
    htform = st.slider("Home Form Points", 0, 15, 7)
    atform = st.slider("Away Form Points", 0, 15, 7)
    diff_pts = htp - atp

with col3:
    st.subheader("Market Odds (B365)")
    b365h = st.number_input("Home Odds", min_value=1.0, value=2.10)
    b365d = st.number_input("Draw Odds", min_value=1.0, value=3.40)
    b365a = st.number_input("Away Odds", min_value=1.0, value=3.80)

# --- PREDICTION LOGIC ---
# 1. Prepare input array for the models
input_features = np.array([[htp, atp, htgd, atgd, diff_pts, htform, atform, b365h, b365d, b365a]])
input_scaled = scaler.transform(input_features)

if st.button("Generate Intelligence Report"):
    st.divider()
    model = models[selected_model_name]
    
    # Get primary prediction and probabilities
    pred = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    # In our Binary Target: 0 = NH (No Win), 1 = H (Home Win)
    home_win_prob = probs[1] if 'H' in model.classes_ else probs[0]

    # --- HANDLING THE 10 QUESTIONS ---
    if "1." in prediction_type:
        result = "HOME WIN" if pred == 'H' else "DRAW or AWAY WIN"
        st.header(f"Result: {result}")
        
    elif "2." in prediction_type:
        st.metric("Win Confidence", f"{home_win_prob*100:.1f}%")
        st.progress(home_win_prob)

    elif "3." in prediction_type:
        if htp < atp and pred == 'H':
            st.warning("⚠️ UPSET ALERT: The underdog is predicted to win at home!")
        else:
            st.success("No significant upset detected.")

    elif "4." in prediction_type:
        if home_win_prob > 0.70:
            st.success("✅ SAFE BET: High confidence in a Home Win.")
        elif home_win_prob < 0.30:
            st.success("✅ SAFE BET: High confidence against a Home Win.")
        else:
            st.info("High Risk: Confidence below 70%.")

    elif "5." in prediction_type:
        # Heuristic: High goal diff and high points suggest goals
        if htgd > 10 and atgd > 10:
            st.write("🔥 High Probability of 'Over 2.5 Goals'. Both teams are scoring machines.")
        else:
            st.write("Neutral Goal Outlook.")

    elif "6." in prediction_type:
        if htgd < 0 and atgd < 0:
            st.write("🛡️ Defensive Battle: Low goal-scoring history detected.")
        else:
            st.write("Standard match tempo.")

    elif "7." in prediction_type:
        st.write(f"Home Form: {htform} pts vs Away Form: {atform} pts")
        if htform > atform + 3:
            st.write("Home team has significant momentum.")
        elif atform > htform + 3:
            st.write("Away team is in superior form.")

    elif "8." in prediction_type:
        st.write(f"League Table Points Gap: {abs(diff_pts)}")
        if abs(diff_pts) > 15:
            st.write("Massive quality gap. The favorite should dominate.")

    elif "9." in prediction_type:
        st.subheader("The Consensus Engine")
        consensus = []
        for name, m in models.items():
            p = m.predict(input_scaled)[0]
            consensus.append(p)
            st.write(f"{name}: {'Win' if p == 'H' else 'No Win'}")
        
    elif "10." in prediction_type:
        if hasattr(model, 'coef_'):
            st.bar_chart(pd.Series(model.coef_[0], index=["HTP","ATP","HTGD","ATGD","Diff","HForm","AForm","OddsH","OddsD","OddsA"]))
        else:
            st.info("Feature importance is not available for this specific model type.")