# === Prep Block ===


# === Imports ===
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

# === League Code to Friendly Name Mapping ===
LEAGUE_NAME_MAP = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SP1": "La Liga",
    "SP2": "Segunda División",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Jupiler Pro League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "G1": "Super League Greece",
    "T1": "Turkish Super Lig",
    "EC": "Champions League",
}

# Reverse mapping: Friendly name ➔ Code
LEAGUE_CODE_MAP = {v: k for k, v in LEAGUE_NAME_MAP.items()}

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Load and Combine Excel Files ===
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
        combined_df = pd.concat([combined_df, temp_df])

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



# === Train Block ===


# === Train Random Forest and Ridge Models ===
features = [
    'HomeTeam', 'AwayTeam', 'Div', 'Month', 'Weekday', 'Hour',
    'ImpH', 'ImpD', 'ImpA', 'Avg>2.5', 'Avg<2.5',
    'HomeTeam_HomeForm', 'AwayTeam_AwayForm',
    'home_GF_rolling5', 'home_GA_rolling5',
    'away_GF_rolling5', 'away_GA_rolling5'
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

def predict_match_with_xg(home_team, away_team, div, month, weekday, hour,
                           imp_h, imp_d, imp_a, over25, under25,
                           home_form, away_form,
                           home_GF_roll5, home_GA_roll5,
                           away_GF_roll5, away_GA_roll5,
                           trained_model, training_columns,
                           home_xg_model, away_xg_model, xg_training_columns):

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

    input_encoded = pd.get_dummies(input_df)
    missing_cols = [col for col in training_columns if col not in input_encoded.columns]
    input_encoded = pd.concat([input_encoded, pd.DataFrame(0, index=input_encoded.index, columns=missing_cols)], axis=1)
    input_encoded = input_encoded[training_columns]

    xg_input_encoded = pd.get_dummies(input_df)
    missing_xg_cols = [col for col in xg_training_columns if col not in xg_input_encoded.columns]
    xg_input_encoded = pd.concat([xg_input_encoded, pd.DataFrame(0, index=xg_input_encoded.index, columns=missing_xg_cols)], axis=1)
    xg_input_encoded = xg_input_encoded[xg_training_columns]

    pred = trained_model.predict(input_encoded)[0]
    prob = trained_model.predict_proba(input_encoded)[0]
    confidence = dict(zip(trained_model.classes_, prob))

    home_xg = home_xg_model.predict(xg_input_encoded)[0]
    away_xg = away_xg_model.predict(xg_input_encoded)[0]

    print(f"\n📢 Prediction: {pred}")
    print("Confidence Scores:")
    for k in ['H', 'D', 'A']:
        print(f"  {k}: {confidence.get(k, 0.0):.2%}")
    print(f"\n⚽ Expected Goals:\n  {home_team}: {home_xg:.2f} xG\n  {away_team}: {away_xg:.2f} xG")

    return pred, confidence, home_xg, away_xg

def predict_match_auto_full(
    home_team, away_team, dt,
    home_odds, draw_odds, away_odds,
    trained_model, training_columns,
    home_xg_model, away_xg_model, xg_training_columns,
    df
):
    recent_match = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('Datetime', ascending=False).head(1)
    div = recent_match['Div'].values[0] if not recent_match.empty else 'P1'

    row = df[
        (df['HomeTeam'] == home_team) & 
        (df['AwayTeam'] == away_team)
    ].sort_values('Datetime', ascending=False).head(1)
    over25 = row['Avg>2.5'].values[0] if not row.empty else df['Avg>2.5'].mean()
    under25 = row['Avg<2.5'].values[0] if not row.empty else df['Avg<2.5'].mean()

    home_form = get_recent_form(home_team, is_home=True, df=df, current_date=dt)
    away_form = get_recent_form(away_team, is_home=False, df=df, current_date=dt)

    def get_gf_ga(team, is_home):
        team_matches = df[
            ((df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)) &
            (df['Datetime'] < dt)
        ].sort_values('Datetime', ascending=False).head(5)
        if is_home:
            gf = team_matches['FTHG']
            ga = team_matches['FTAG']
        else:
            gf = team_matches['FTAG']
            ga = team_matches['FTHG']
        return np.mean(gf), np.mean(ga)

    home_GF_roll5, home_GA_roll5 = get_gf_ga(home_team, is_home=True)
    away_GF_roll5, away_GA_roll5 = get_gf_ga(away_team, is_home=False)

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
        trained_model, training_columns,
        home_xg_model, away_xg_model, xg_training_columns
    )
# === Predictions ===

predict_match_auto_full(
    home_team='Casa Pia',
    away_team='Estoril',
    dt=pd.Timestamp('2025-04-29 21:30'),
    home_odds=2.45,
    draw_odds=2.95,
    away_odds=3.25,
    trained_model=rf_model,
    training_columns=training_columns,
    home_xg_model=home_xg_model,
    away_xg_model=away_xg_model,
    xg_training_columns=xg_training_columns,
    df=combined_df
)

# === Streamlit App ===
import streamlit as st

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
    'Segunda Division': 'SP2',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Bundesliga 1': 'D1',
    'Bundesliga 2': 'D2',
    'Ligue 1': 'F1',
    'Ligue 2': 'F2',
    'Primeira Liga': 'P1',
    'Eredivisie': 'N1',
    'Scottish Premier': 'SC0',
    'Belgian First Division': 'B1',
    'Greek Super League': 'G1',
    'Turkish Super Lig': 'T1',
    'Swiss Super League': 'EC',
}

# Select league
selected_league_name = st.selectbox('Select League:', list(LEAGUE_CODE_MAP.keys()))
selected_league_code = LEAGUE_CODE_MAP[selected_league_name]

# Filter team list based on league
filtered_df = combined_df[combined_df['Div'] == selected_league_code]
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

# Predict button
if st.button("Predict Match"):
    if home_team and away_team:
        dt = pd.Timestamp(f"{match_date} {match_time}")
        # Call your prediction function
        pred, confidence, home_xg, away_xg = predict_match_auto_full(
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
            df=combined_df
        )

        st.subheader("📢 Prediction Result")
        st.write(f"**Predicted Result:** {pred}")

        st.subheader("🔎 Confidence Levels")
        st.write(f"Home Win: {confidence['H']:.2%}")
        st.write(f"Draw: {confidence['D']:.2%}")
        st.write(f"Away Win: {confidence['A']:.2%}")

        st.subheader("⚽ Expected Goals (xG)")
        st.write(f"{home_team}: {home_xg:.2f} xG")
        st.write(f"{away_team}: {away_xg:.2f} xG")
    else:
        st.error("Please enter both Home and Away teams.")
