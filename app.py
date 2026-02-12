import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="EPL Smart Predictor v2.1", layout="wide", page_icon="⚽")

# --- LOAD ASSETS (MODELS, SCALER, AND TEAM DB) ---
@st.cache_resource
def load_assets():
    # 1. Load models and scaler
    try:
        scaler = joblib.load('models/scaler.pkl')
        models = {
            "KNN": joblib.load('models/knn_model.pkl'),
            "Logistic Regression": joblib.load('models/logistic_model.pkl'),
            "SVM": joblib.load('models/svm_model.pkl'),
            "Naive Bayes": joblib.load('models/nb_model.pkl'),
            "Neural Network": joblib.load('models/nn_model.pkl')
        }
        # 2. Load Current Season Database and index by Team name for fast lookup
        team_db = pd.read_csv('current_season_db.csv').set_index('Team')
        return scaler, models, team_db
    except FileNotFoundError as e:
        st.error(f"CRITICAL ERROR: Missing essential file. {e}")
        st.stop()

scaler, models, team_db = load_assets()
team_list = team_db.index.tolist()

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🤖 AI Control Panel")
selected_model_name = st.sidebar.selectbox("Select AI Brain", list(models.keys()), index=1) # Default to Logistic Reg

# --- ALL 10 MODES RESTORED ---
prediction_type = st.sidebar.selectbox("Analytics Mode", [
    "1. Straight Win/Loss",
    "2. Win Probability %",
    "3. Upset Alert",
    "4. 'Safe Bet' Filter",
    "5. Goal Fest Prediction",
    "6. Defensive Battle",
    "7. Form Check",
    "8. Points Gap Analysis",
    "9. Algorithm Comparison (Consensus)",
    "10. Feature Importance"
])
st.sidebar.markdown("---")
st.sidebar.info("Note: Betting odds must still be entered manually as they change for every specific matchup.")

# --- MAIN INTERFACE: MATCH SETUP ---
st.title("⚽ EPL Smart Prediction Engine")
st.markdown(f"Powered by: **{selected_model_name}**")

# --- STEP 1: SELECT TEAMS ---
st.subheader("1. Select Matchup")
col_home, col_vs, col_away = st.columns([2,1,2])
with col_home:
    home_team = st.selectbox("Home Team", team_list, index=0)
with col_vs:
    st.markdown("<h2 style='text-align: center; padding-top: 20px;'>VS</h2>", unsafe_allow_html=True)
with col_away:
    away_team = st.selectbox("Away Team", team_list, index=1)

if home_team == away_team:
    st.error("⚠️ Please select two different teams.")
    st.stop()

# --- STEP 2: AUTOMATIC DATA RETRIEVAL ---
# Look up stats from the loaded CSV database
h_stats = team_db.loc[home_team]
a_stats = team_db.loc[away_team]

# Assign variables needed for the model
htp = h_stats['TotalPoints']
atp = a_stats['TotalPoints']
htgd = h_stats['GoalDifference']
atgd = a_stats['GoalDifference']
htform = h_stats['FormPointsLast5']
atform = a_stats['FormPointsLast5']
diff_pts = htp - atp

# --- STEP 3: DISPLAY THE "REASONS" (PRE-MATCH STATS) ---
st.subheader("2. Pre-Match Intelligence Briefing")
st.markdown("The AI is analyzing the following current stats:")

# Create a neat comparison table showing the "Reasons"
match_stats_df = pd.DataFrame({
    "Metric": ["Total Points (Season)", "Goal Difference", "Form (Last 5 Games)"],
    f"{home_team} (Home)": [htp, htgd, htform],
    f"{away_team} (Away)": [atp, atgd, atform]
}).set_index("Metric")

st.table(match_stats_df)

# --- STEP 4: MARKET ODDS (Must remain manual) ---
st.subheader("3. Market Context (Live Odds)")
st.caption("Enter current bookmaker odds for this specific match.")
o1, o2, o3 = st.columns(3)
b365h = o1.number_input(f"{home_team} To Win", min_value=1.01, value=2.10, step=0.1)
b365d = o2.number_input("Draw", min_value=1.01, value=3.50, step=0.1)
b365a = o3.number_input(f"{away_team} To Win", min_value=1.01, value=4.20, step=0.1)


# --- RUN PREDICTION ---
st.markdown("---")
if st.button("🚀 Run Predictive Analysis", type="primary"):
    
    # 1. Prepare input array
    input_features = np.array([[htp, atp, htgd, atgd, diff_pts, htform, atform, b365h, b365d, b365a]])
    input_scaled = scaler.transform(input_features)
    
    # 2. Get Prediction based on selected model
    model = models[selected_model_name]
    pred = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]
    # Ensure we get the probability for the 'H' (Home Win) class
    # Some models might order classes differently, this ensures we get the right index
    home_win_idx = np.where(model.classes_ == 'H')[0][0]
    home_win_prob = probs[home_win_idx]

    # 3. Display Results based on selected mode
    st.header("Analytics Report")

    if "1." in prediction_type:
        if pred == 'H':
            st.success(f"**Prediction: HOME WIN ({home_team})**")
        else:
            st.error(f"**Prediction: NOT A HOME WIN (Draw or {away_team})**")

    elif "2." in prediction_type:
        st.metric(f"{home_team} Win Probability", f"{home_win_prob*100:.1f}%")
        st.progress(home_win_prob)
        if home_win_prob > 0.5:
            st.info(f"The model favors {home_team}.")
        else:
            st.info(f"The model favors a Draw or {away_team}.")

    elif "3." in prediction_type:
        # Upset if Home team has fewer points but is predicted to win
        if htp < atp and pred == 'H':
            st.warning(f"⚠️ UPSET ALERT! {home_team} are lower in the table but predicted to win.")
        else:
            st.info("No upset condition detected based on league table positions.")

    elif "4." in prediction_type:
        if home_win_prob > 0.70:
            st.success(f"✅ SAFE BET: High confidence (>70%) in {home_team} winning.")
        elif home_win_prob < 0.30:
            st.success(f"✅ SAFE BET: High confidence (>70%) against {home_team} winning.")
        else:
            st.warning("Analyze carefully: Confidence is moderate. No 'Safe Bet' flag.")

    # --- RESTORED MODES 5, 6, 7, 8 ---
    elif "5." in prediction_type:
        st.subheader("Goal Fest Analysis")
        # Heuristic: If both teams have high positive Goal Difference
        if htgd > 15 and atgd > 15:
             st.write("🔥 **High Goal Potential:** Both teams have very strong attacking records this season.")
        elif htgd > 5 and atgd > 5:
             st.write("Moderate Goal Potential: Both teams have positive goal differences.")
        else:
             st.write("Neutral outlook for a high-scoring game based on season stats.")

    elif "6." in prediction_type:
        st.subheader("Defensive Battle Analysis")
        # Heuristic: If both teams have negative or neutral Goal Difference
        if htgd <= 0 and atgd <= 0:
            st.write("🛡️ **Defensive Grind:** Both teams have negative or neutral goal differences. Expect a tight, low-scoring affair.")
        else:
            st.write("Stats do not indicate a purely defensive struggle.")

    elif "7." in prediction_type:
        st.subheader("Momentum & Form Check")
        st.write(f"{home_team} Last 5 Games: **{htform} pts**")
        st.write(f"{away_team} Last 5 Games: **{atform} pts**")
        form_diff = htform - atform
        if form_diff > 3:
             st.success(f"Significant momentum advantage for {home_team}.")
        elif form_diff < -3:
             st.error(f"Significant momentum advantage for {away_team}.")
        else:
             st.info("Both teams are in relatively similar recent form.")

    elif "8." in prediction_type:
        st.subheader("League Standings Gap")
        st.write(f"Points Difference: **{abs(diff_pts)}**")
        if abs(diff_pts) > 20:
            st.write("Massive quality gap detected between top and bottom tier teams.")
        elif abs(diff_pts) > 10:
            st.write("Significant gap in league standing.")
        else:
            st.write("These teams are relatively close in the league table.")

    # --- EXISTING MODE 9 ---
    elif "9." in prediction_type:
        st.subheader("Consensus Engine (All Models)")
        col_con1, col_con2 = st.columns(2)
        home_votes = 0
        with col_con1:
            for name, m in models.items():
                p = m.predict(input_scaled)[0]
                if p == 'H': home_votes += 1
                st.write(f"**{name}**: {'🔴 Home Win' if p == 'H' else '🔵 Not Home Win'}")
        with col_con2:
             st.metric("Consensus Agreement", f"{home_votes}/5 Models predict Home Win")

    # --- RESTORED MODE 10 ---
    elif "10." in prediction_type:
        st.subheader("Feature Importance (Why?)")
        # Feature importance only works for linear models like Logistic Regression or Linear SVM
        if hasattr(model, 'coef_'):
            st.write("Positive values favor Home Win, Negative favor Not Home Win.")
            # Create a chart mapping coefficients to feature names
            feat_importance = pd.Series(model.coef_[0], index=["HTP","ATP","HTGD","ATGD","DiffPts","HTForm","ATForm","OddsH","OddsD","OddsA"])
            st.bar_chart(feat_importance)
        else:
            st.warning(f"Sorry, 'Feature Importance' is not available for the **{selected_model_name}** model. Please select Logistic Regression to view this chart.")
