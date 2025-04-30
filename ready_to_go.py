# === Prep Block ===


# === Imports ===
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
import os  # for checking if .pkl files exist
skip_training = os.path.exists("rf_model.pkl")



# === League Code to Friendly Name Mapping ===
LEAGUE_NAME_MAP = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "P1": "Liga Portugal",
    "B1": "Jupiler Pro League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "G1": "Super League Greece",
    "T1": "Turkish Super Lig",
    "EC": "National League",
}

# Reverse mapping: Friendly name ➔ Code
LEAGUE_CODE_MAP = {v: k for k, v in LEAGUE_NAME_MAP.items()}

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Load and Combine Excel Files === from https://www.football-data.co.uk/downloadm.php
import streamlit as st

@st.cache_data
def load_combined_data():
    excel_files = [
        'all-euro-data-2023-2024.xlsx',
        'all-euro-data-2024-2025.xlsx'
    ]

    combined_df = pd.DataFrame()

    for file in excel_files:
        excel = pd.ExcelFile(file)
        for sheet_name in excel.sheet_names:
            temp_df = excel.parse(sheet_name)
            temp_df['League'] = sheet_name
            temp_df['Season'] = file.split('/')[-1].replace('.xlsx', '')

            for col in temp_df.columns:
                try:
                    temp_df[col] = pd.to_numeric(temp_df[col])
                except (ValueError, TypeError):
                    continue

            # Add BTTS column: 1 if both teams scored
            temp_df['BTTS'] = ((temp_df['FTHG'] > 0) & (temp_df['FTAG'] > 0)).astype(int)

            combined_df = pd.concat([combined_df, temp_df])

    return combined_df.reset_index(drop=True)

# === Basic Cleaning ===
combined_df.dropna(axis=1, how='all', inplace=True)
combined_df.dropna(axis=0, how='all', inplace=True)
combined_df['Date'] = pd.to_datetime(combined_df['Date'] / 1e9, unit='s')
combined_df['Datetime'] = pd.to_datetime(combined_df['Date'].astype(str) + ' ' + combined_df['Time'].astype(str), errors='coerce', dayfirst=True)
combined_df = combined_df.sort_values(by='Datetime').reset_index(drop=True)

# === Feature Engineering ===
combined_df['Month'] = combined_df['Datetime'].dt.month
combined_df['Weekday'] = combined_df['Datetime'].dt.day_name()
combined_df['Hour'] = combined_df['Datetime'].dt.hour

for col in ['AvgH', 'AvgD', 'AvgA']:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df['ImpH'] = 1 / combined_df['AvgH']
combined_df['ImpD'] = 1 / combined_df['AvgD']
combined_df['ImpA'] = 1 / combined_df['AvgA']
sum_probs = combined_df['ImpH'] + combined_df['ImpD'] + combined_df['ImpA']
combined_df['ImpH'] /= sum_probs
combined_df['ImpD'] /= sum_probs
combined_df['ImpA'] /= sum_probs

# === Rolling Form (Points) ===
home_form = defaultdict(lambda: deque(maxlen=5))
away_form = defaultdict(lambda: deque(maxlen=5))
points_map = {'H': (3, 0), 'D': (1, 1), 'A': (0, 3)}

combined_df['HomeTeam_HomeForm'] = np.nan
combined_df['AwayTeam_AwayForm'] = np.nan

for idx, row in combined_df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    result = row['FTR']

    combined_df.at[idx, 'HomeTeam_HomeForm'] = np.mean(home_form[home_team]) if home_form[home_team] else np.nan
    combined_df.at[idx, 'AwayTeam_AwayForm'] = np.mean(away_form[away_team]) if away_form[away_team] else np.nan

    if result in points_map:
        home_pts, away_pts = points_map[result]
        home_form[home_team].append(home_pts)
        away_form[away_team].append(away_pts)

# === Rolling GF/GA Stats ===
rolling_stats = {
    "home_GF": defaultdict(lambda: deque(maxlen=5)),
    "home_GA": defaultdict(lambda: deque(maxlen=5)),
    "away_GF": defaultdict(lambda: deque(maxlen=5)),
    "away_GA": defaultdict(lambda: deque(maxlen=5)),
}

for key in rolling_stats:
    combined_df[f"{key}_rolling5"] = np.nan

for idx, row in combined_df.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]
    fthg = row["FTHG"]
    ftag = row["FTAG"]

    combined_df.at[idx, "home_GF_rolling5"] = np.mean(rolling_stats["home_GF"][home]) if rolling_stats["home_GF"][home] else np.nan
    combined_df.at[idx, "home_GA_rolling5"] = np.mean(rolling_stats["home_GA"][home]) if rolling_stats["home_GA"][home] else np.nan
    combined_df.at[idx, "away_GF_rolling5"] = np.mean(rolling_stats["away_GF"][away]) if rolling_stats["away_GF"][away] else np.nan
    combined_df.at[idx, "away_GA_rolling5"] = np.mean(rolling_stats["away_GA"][away]) if rolling_stats["away_GA"][away] else np.nan

    rolling_stats["home_GF"][home].append(fthg)
    rolling_stats["home_GA"][home].append(ftag)
    rolling_stats["away_GF"][away].append(ftag)
    rolling_stats["away_GA"][away].append(fthg)

# === Create BTTS (Both Teams To Score) column ===
combined_df['BTTS'] = ((combined_df['FTHG'] > 0) & (combined_df['FTAG'] > 0)).astype(int)

import os
import joblib

# === Attempt to load saved models and training structures ===
if (
    os.path.exists("rf_model.pkl") and
    os.path.exists("home_xg_model.pkl") and
    os.path.exists("away_xg_model.pkl") and
    os.path.exists("btts_model.pkl") and
    os.path.exists("training_columns.pkl") and
    os.path.exists("xg_training_columns.pkl") and
    os.path.exists("btts_training_columns.pkl") and
    os.path.exists("results.pkl")  # ✅ Add this check

):
    rf_model = joblib.load("rf_model.pkl")
    home_xg_model = joblib.load("home_xg_model.pkl")
    away_xg_model = joblib.load("away_xg_model.pkl")
    btts_model = joblib.load("btts_model.pkl")
    training_columns = joblib.load("training_columns.pkl")
    xg_training_columns = joblib.load("xg_training_columns.pkl")
    btts_training_columns = joblib.load("btts_training_columns.pkl")
    results = joblib.load("results.pkl")  # ✅ Load results here
    skip_training = True
    print("✅ Models loaded from disk.")

else:
    skip_training = False

# === Train Block ===

if not skip_training:

    # === Train Random Forest and Ridge Models ===
    features = [
        'HomeTeam', 'AwayTeam', 'Div', 'Month', 'Weekday', 'Hour',
        'ImpH', 'ImpD', 'ImpA', 'Avg>2.5', 'Avg<2.5',
        'HomeTeam_HomeForm', 'AwayTeam_AwayForm',
        'home_GF_rolling5', 'home_GA_rolling5',
        'away_GF_rolling5', 'away_GA_rolling5',
        'HR', 'AR'  # 🔴 Red cards
    ]

    combined = combined_df[features + ['FTR']].dropna()
    X = combined.drop(columns='FTR')
    y = combined['FTR']
    X_encoded = pd.get_dummies(X)


    # Match outcome model
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_encoded, y)

    # Save column structures
    training_columns = X_encoded.columns
    xg_training_columns = list(X_encoded.columns)

    # xG models
    home_xg_model = Ridge().fit(X_encoded, combined_df.loc[X_encoded.index, 'FTHG'])
    away_xg_model = Ridge().fit(X_encoded, combined_df.loc[X_encoded.index, 'FTAG'])

    # === Train BTTS Model ===
    btts_df = combined_df.dropna(subset=[
        'FTHG', 'FTAG', 'BTTS',  # outcome columns
        'HomeTeam_HomeForm', 'AwayTeam_AwayForm',
        'home_GF_rolling5', 'home_GA_rolling5',
        'away_GF_rolling5', 'away_GA_rolling5'
    ])

    X_btts = pd.get_dummies(btts_df[[
        'HomeTeam', 'AwayTeam', 'Div', 'Month', 'Weekday', 'Hour',
        'ImpH', 'ImpD', 'ImpA',
        'HomeTeam_HomeForm', 'AwayTeam_AwayForm',
        'home_GF_rolling5', 'home_GA_rolling5',
        'away_GF_rolling5', 'away_GA_rolling5'
    ]])

    y_btts = btts_df['BTTS']

    btts_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    btts_model.fit(X_btts, y_btts)

    # Save structure
    btts_training_columns = X_btts.columns

    # === Per-League Accuracy Evaluation ===
    from sklearn.metrics import classification_report

    target_divs = combined_df['Div'].dropna().unique()
    results = {}

    print("📊 Model Performance by League\n")

    for div in target_divs:
        df_div = combined_df[combined_df['Div'] == div].copy()
        df_div = df_div.dropna(subset=features + ['FTR'])

        if len(df_div) < 30:
            continue

        X_div = df_div[features]
        y_div = df_div['FTR']
        X_encoded_div = pd.get_dummies(X_div)

        # Efficiently align columns
        X_encoded_div = X_encoded_div.reindex(columns=training_columns, fill_value=0)

        X_train, X_test, y_train, y_test = train_test_split(
        X_encoded_div, y_div, test_size=0.2, stratify=y_div, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)


        print(f"🔍 Training on Division: {div}")
        print(f"✅ Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        results[div] = acc

    # === Visualize Per-Division Accuracy ===
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))  # Sort by accuracy

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_results.keys(), sorted_results.values(), color='skyblue')
    plt.title('Random Forest Accuracy by League (Ascending)')
    plt.xlabel('League')
    plt.ylabel('Accuracy')
    plt.ylim(0.3, 0.7)
    plt.axhline(0.44, color='red', linestyle='--', label='Baseline (All Leagues)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    joblib.dump(results, "results.pkl")  # Save per-league accuracy dictionary

    import joblib 

    joblib.dump(rf_model, "rf_model.pkl")     # Save Random Forest model
    joblib.dump(home_xg_model, "home_xg_model.pkl")     # Save xG models
    joblib.dump(away_xg_model, "away_xg_model.pkl")
    joblib.dump(btts_model, "btts_model.pkl")    # Save BTTS model
    joblib.dump(training_columns, "training_columns.pkl")     # Save training column structures
    joblib.dump(xg_training_columns, "xg_training_columns.pkl")
    joblib.dump(btts_training_columns, "btts_training_columns.pkl")
    joblib.dump(results, "results.pkl")  # ✅ Save per-league accuracy

# === Functions ===
# === Support Functions ===
def get_recent_form(team_name, is_home, df, current_date):
    mask = ((df['HomeTeam'] == team_name) if is_home else (df['AwayTeam'] == team_name)) & (df['Datetime'] < current_date)
    recent = df[mask].sort_values('Datetime', ascending=False).head(5)
    points = []
    for _, row in recent.iterrows():
        if is_home:
            points.append({'H': 3, 'D': 1, 'A': 0}.get(row['FTR'], 0))
        else:
            points.append({'A': 3, 'D': 1, 'H': 0}.get(row['FTR'], 0))
    return np.mean(points) if points else np.nan

# Here's the updated version of the prediction functions including BTTS prediction.

# Place these in your Functions block, replacing your existing versions of `predict_match_with_xg()` and `predict_match_auto_full()`

def predict_match_with_xg_and_btts(home_team, away_team, div, month, weekday, hour,
                                   imp_h, imp_d, imp_a, over25, under25,
                                   home_form, away_form,
                                   home_GF_roll5, home_GA_roll5,
                                   away_GF_roll5, away_GA_roll5,
                                   trained_model, training_columns,
                                   home_xg_model, away_xg_model, xg_training_columns,
                                   btts_model, btts_training_columns):

    input_df = pd.DataFrame([{
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Div': div,
        'Month': month,
        'Weekday': weekday,
        'Hour': hour,
        'ImpH': imp_h,
        'ImpD': imp_d,
        'ImpA': imp_a,
        'Avg>2.5': over25,
        'Avg<2.5': under25,
        'HomeTeam_HomeForm': home_form,
        'AwayTeam_AwayForm': away_form,
        'home_GF_rolling5': home_GF_roll5,
        'home_GA_rolling5': home_GA_roll5,
        'away_GF_rolling5': away_GF_roll5,
        'away_GA_rolling5': away_GA_roll5
    }])

    # Encode inputs
    input_encoded = pd.get_dummies(input_df)
    for col in training_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[training_columns]

    # Predict result
    pred = trained_model.predict(input_encoded)[0]
    prob = trained_model.predict_proba(input_encoded)[0]
    confidence = dict(zip(trained_model.classes_, prob))

    # Predict xG
    xg_input_encoded = pd.get_dummies(input_df).reindex(columns=xg_training_columns, fill_value=0)
    home_xg = home_xg_model.predict(xg_input_encoded)[0]
    away_xg = away_xg_model.predict(xg_input_encoded)[0]

    # Predict BTTS
    btts_input_encoded = pd.get_dummies(input_df).reindex(columns=btts_training_columns, fill_value=0)
    btts_proba = btts_model.predict_proba(btts_input_encoded)[0][1]

    # Output
    print(f"\n📢 Prediction: {pred}")
    print("Confidence Scores:")
    for k in ['H', 'D', 'A']:
        print(f"  {k}: {confidence.get(k, 0.0):.2%}")
    print(f"\n⚽ Expected Goals:\n  {home_team}: {home_xg:.2f} xG\n  {away_team}: {away_xg:.2f} xG")
    print(f"\n🤝 Both Teams to Score (BTTS): {btts_proba:.2%}")

    return pred, confidence, home_xg, away_xg, btts_proba


def predict_match_auto_full(home_team, away_team, dt,
                             home_odds, draw_odds, away_odds,
                             trained_model, training_columns,
                             home_xg_model, away_xg_model, xg_training_columns,
                             btts_model, btts_training_columns,
                             df):
    recent_match = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('Datetime', ascending=False).head(1)
    div = recent_match['Div'].values[0] if not recent_match.empty else 'P1'

    row = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)].sort_values('Datetime', ascending=False).head(1)
    over25 = row['Avg>2.5'].values[0] if not row.empty else df['Avg>2.5'].mean()
    under25 = row['Avg<2.5'].values[0] if not row.empty else df['Avg<2.5'].mean()

    def get_form(team, is_home):
        recent = df[
            ((df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)) &
            (df['Datetime'] < dt)
        ].sort_values('Datetime', ascending=False).head(5)
        points = []
        for _, r in recent.iterrows():
            outcome = r['FTR']
            points.append({'H': 3, 'D': 1, 'A': 0}.get(outcome, 0) if is_home else {'A': 3, 'D': 1, 'H': 0}.get(outcome, 0))
        return np.mean(points) if points else np.nan

    def get_gf_ga(team, is_home):
        recent = df[
            ((df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)) &
            (df['Datetime'] < dt)
        ].sort_values('Datetime', ascending=False).head(5)
        gf = recent['FTHG'] if is_home else recent['FTAG']
        ga = recent['FTAG'] if is_home else recent['FTHG']
        return gf.mean(), ga.mean()

    home_form = get_form(home_team, is_home=True)
    away_form = get_form(away_team, is_home=False)
    home_GF_roll5, home_GA_roll5 = get_gf_ga(home_team, is_home=True)
    away_GF_roll5, away_GA_roll5 = get_gf_ga(away_team, is_home=False)

    sum_odds = 1 / home_odds + 1 / draw_odds + 1 / away_odds
    imp_h = (1 / home_odds) / sum_odds
    imp_d = (1 / draw_odds) / sum_odds
    imp_a = (1 / away_odds) / sum_odds

    return predict_match_with_xg_and_btts(
        home_team, away_team, div, dt.month, dt.day_name(), dt.hour,
        imp_h, imp_d, imp_a, over25, under25,
        home_form, away_form,
        home_GF_roll5, home_GA_roll5,
        away_GF_roll5, away_GA_roll5,
        trained_model, training_columns,
        home_xg_model, away_xg_model, xg_training_columns,
        btts_model, btts_training_columns
    )


    # --- Rolling red cards ---
    def get_red_cards(team, is_home):
        team_matches = df[
            ((df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)) &
            (df['Datetime'] < dt)
        ].sort_values('Datetime', ascending=False).head(5)

        red_col = 'HR' if is_home else 'AR'
        return np.mean(team_matches[red_col]) if red_col in team_matches else 0

    home_RC = get_red_cards(home_team, is_home=True)
    away_RC = get_red_cards(away_team, is_home=False)

    month = dt.month
    weekday = dt.day_name()
    hour = dt.hour

    # 🎯 Calculate implied probabilities from odds
    imp_h = 1 / home_odds
    imp_d = 1 / draw_odds
    imp_a = 1 / away_odds
    total = imp_h + imp_d + imp_a

    imp_h /= total
    imp_d /= total
    imp_a /= total
    
    print(f"\n🎯 Odds Input:\nHome Odds: {home_odds}\nDraw Odds: {draw_odds}\nAway Odds: {away_odds}")


    return predict_match_with_xg(
        home_team, away_team, div, month, weekday, hour,
        imp_h, imp_d, imp_a, over25, under25,
        home_form, away_form,
        home_GF_roll5, home_GA_roll5,
        away_GF_roll5, away_GA_roll5,
        home_RC, away_RC,  # ✅ Add these two
        trained_model, training_columns,
        home_xg_model, away_xg_model, xg_training_columns
    )



# === Streamlit App and Google Spreadsheet===
import gspread
import streamlit as st
from datetime import datetime

def ensure_sheet_headers(sheet_name, creds_path="bet25-458323-5032ba07639b.json"):
    headers = [
        "Timestamp", "League", "Home Team", "Away Team", "DateTime",
        "Odds (H)", "Odds (D)", "Odds (A)",
        "Prediction", "Conf. H", "Conf. D", "Conf. A",
        "xG Home", "xG Away", "BTTS Prob"
    ]

    gc = gspread.service_account(filename=creds_path)
    sh = gc.open(sheet_name)
    worksheet = sh.sheet1

    existing_headers = worksheet.row_values(1)
    if existing_headers != headers:
        worksheet.delete_rows(1)
        worksheet.insert_row(headers, index=1)


from datetime import datetime

def log_prediction_to_sheet(sheet_name, row_data, creds_path="bet25-458323-5032ba07639b.json"):
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open(sheet_name)
    worksheet = sh.sheet1

    # Define column headers (exact order must match row_data)
    headers = [
        "Timestamp", "League", "Home Team", "Away Team", "DateTime",
        "Odds (H)", "Odds (D)", "Odds (A)",
        "Prediction", "Conf. H", "Conf. D", "Conf. A",
        "xG Home", "xG Away", "BTTS Prob"
    ]

    # Add headers only if sheet is empty
    if len(worksheet.get_all_values()) == 0:
        worksheet.append_row(headers)

    # Timestamp for the first column
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_with_time = [timestamp] + [str(val).replace(",", ".") for val in row_data]

    worksheet.append_row(row_with_time)



st.title("⚽ Football Match Predictor App")

# User inputs
# === 1. League → Home Team → Away Team Dynamic Dropdowns ===

# === Dropdowns for League, Home Team, Away Team ===

# Mapping readable league names to internal league codes
LEAGUE_CODE_MAP = {
    'Premier League': 'E0',
    'Championship': 'E1',
    'League One': 'E2',
    'League Two': 'E3',
    'La Liga': 'SP1',
    'La Liga 2': 'SP2',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga 1': 'D1',
    'Bundesliga 2': 'D2',
    'Ligue 1': 'F1',
    'Ligue 2': 'F2',
    'Liga Portugal': 'P1',
    'Eredivisie': 'N1',
    'Scottish Premier': 'SC0',
    'Belgian First Division': 'B1',
    'Greek Super League': 'G1',
    'Turkish Super Lig': 'T1',
    'National League': 'EC',
}

# Select league
# Add "All Leagues" option to the top of the dropdown
LEAGUE_NAMES_WITH_ALL = ["All Leagues"] + list(LEAGUE_CODE_MAP.keys())
selected_league_name = st.selectbox('Select League:', LEAGUE_NAMES_WITH_ALL)

# Returns None if "All Leagues" is selected
selected_league_code = LEAGUE_CODE_MAP.get(selected_league_name, None)

# Display accuracy for selected league
if selected_league_code in results:
    league_accuracy = results[selected_league_code]
    st.markdown(f"🧠 **Model Accuracy for {selected_league_name}:** {league_accuracy:.2%}")
    
else:
    st.markdown("🧠 Accuracy data not available for this league.")
    



# Filter team list based on league
if selected_league_code:
    # Filter by selected league
    filtered_df = combined_df[combined_df['Div'] == selected_league_code]
else:
    # All leagues
    filtered_df = combined_df

all_teams = sorted(pd.unique(filtered_df[['HomeTeam', 'AwayTeam']].values.ravel()))

# Home Team selection (filtered)
home_team = st.selectbox('Select Home Team:', all_teams, key='home_team')

# Away Team selection (filtered and excluding the Home Team)
away_team_options = [team for team in all_teams if team != home_team]
away_team = st.selectbox('Select Away Team:', away_team_options, key='away_team')

match_date = st.date_input("Select Match Date:")
match_time = st.time_input("Select Match Time:")

home_odds = st.number_input("Enter Home Win Odds", value=2.50)
draw_odds = st.number_input("Enter Draw Odds", value=3.00)
away_odds = st.number_input("Enter Away Win Odds", value=3.00)


import gspread

def log_prediction_to_sheet(sheet_name, row_data, creds_path="bet25-458323-5032ba07639b.json"):
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open("Prediction Logs")
    worksheet = sh.sheet1  # You can use a named sheet if needed: sh.worksheet("Sheet1")

    worksheet.append_row(row_data)



# Predict button
if st.button("Predict Match"):
    if home_team and away_team:
        dt = pd.Timestamp(f"{match_date} {match_time}")
        # Call your prediction function
        pred, confidence, home_xg, away_xg, btts_prob = predict_match_auto_full(
            home_team=home_team,
            away_team=away_team,
            dt=dt,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            trained_model=rf_model,
            training_columns=training_columns,
            home_xg_model=home_xg_model,
            away_xg_model=away_xg_model,
            xg_training_columns=xg_training_columns,
            df=combined_df,
            btts_model=btts_model,
            btts_training_columns=btts_training_columns
        )

        ###st.subheader("📢 Prediction Result")###
        ###st.write(f"**Predicted Result:** {pred}")###

        st.subheader("🔎 Confidence Levels")
        st.write(f"Home Win: {confidence['H']:.2%}")
        st.write(f"Draw: {confidence['D']:.2%}")
        st.write(f"Away Win: {confidence['A']:.2%}")

        st.subheader("⚽ Expected Goals (xG)")
        st.write(f"{home_team}: {home_xg:.2f} xG")
        st.write(f"{away_team}: {away_xg:.2f} xG")
        
        st.subheader("🎯 BTTS (Both Teams To Score)")
        
        # Textual label based on probability
        if btts_prob > 0.65:
            confidence_label = "High"
        elif btts_prob > 0.45:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        st.write(f"Chance: **{btts_prob * 100:.1f}%** ({confidence_label} confidence)")
        st.progress(min(int(btts_prob * 100), 100), text="BTTS Probability")


        # Prepare row for logging (aligned with headers)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_data = [
            timestamp,
            selected_league_name,
            home_team,
            away_team,
            str(dt),
            home_odds,
            draw_odds,
            away_odds,
            pred,
            confidence['H'],  # instead of f"{confidence['H']:.2%}"
            confidence['D'],
            confidence['A'],
            home_xg,
            away_xg,
            btts_prob
        ]


        try:
            ensure_sheet_headers("Prediction Logs")
            log_prediction_to_sheet("Prediction Logs", row_data)
            st.success("✅ Prediction logged to Google Sheets.")
        except Exception as e:
            st.warning(f"⚠️ Logging failed: {e}")
        
    else:
        st.error("Please enter both Home and Away teams.")

    def ensure_sheet_headers(sheet_name, creds_path="bet25-458323-5032ba07639b.json"):
        import gspread
        gc = gspread.service_account(filename=creds_path)
        sh = gc.open(sheet_name)
        worksheet = sh.sheet1

        expected_headers = [
            "Timestamp", "League", "Home Team", "Away Team",
            "Home Odds", "Draw Odds", "Away Odds",
            "Conf. Home", "Conf. Draw", "Conf. Away",
            "xG Home", "xG Away", "BTTS Probability"
        ]

        current_headers = worksheet.row_values(1)
        if current_headers != expected_headers:
            worksheet.insert_row(expected_headers, index=1)   
