"""Lead-to-Counselor assignment app (baseline).

Given a new lead's attributes, this app predicts which counselor is most likely
to convert that lead into an enrolled student, using a Random Forest trained on
historical lead/conversion data.
"""

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "Lead Conversion Data.xlsx"
FEATURES = ["College", "Program Level", "Program of Study", "Counselor", "Counselor Level"]


@st.cache_data(show_spinner="Loading lead data…")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    # calamine is much faster than openpyxl on large workbooks; fall back if absent.
    try:
        df = pd.read_excel(path, engine="calamine")
    except Exception:
        df = pd.read_excel(path)
    # Target: did this lead convert into a "Student"?
    df["Converted"] = (
        df["Record Type"].astype(str).str.strip().str.lower() == "student"
    ).astype(int)
    return df


@st.cache_resource(show_spinner="Training model…")
def train_model(path: str = DATA_PATH) -> Pipeline:
    df = load_data(path)
    model_df = df[FEATURES + ["Converted"]].dropna()
    X, y = model_df[FEATURES], model_df["Converted"]

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    return pipeline


df = load_data()
pipeline = train_model()

# Show the time range of the underlying data (column is "Lead Created On").
if "Lead Created On" in df.columns:
    created = pd.to_datetime(df["Lead Created On"], errors="coerce")
    if created.notna().any():
        st.markdown(f"**Data Time Range:** {created.min().date()} to {created.max().date()}")

st.title("Lead to Counselor Assignment App")
st.write("Input new lead details to automatically assign the best counselor.")

college = st.selectbox("College", sorted(df["College"].dropna().unique()))
program_level = st.selectbox("Program Level", sorted(df["Program Level"].dropna().unique()))
program_of_study = st.selectbox(
    "Program of Study", sorted(df["Program of Study"].dropna().unique())
)

if st.button("Assign Counselor"):
    # Build one candidate row per (Counselor, Counselor Level) and score them all at once.
    counselors = (
        df[["Counselor", "Counselor Level"]].dropna().drop_duplicates().reset_index(drop=True)
    )
    candidates = counselors.copy()
    candidates["College"] = college
    candidates["Program Level"] = program_level
    candidates["Program of Study"] = program_of_study

    pos = list(pipeline.classes_).index(1) if 1 in pipeline.classes_ else 0
    counselors["Conversion_Probability"] = pipeline.predict_proba(candidates[FEATURES])[:, pos]

    # Historical volume per counselor for context.
    stats = df.groupby("Counselor")["Converted"].agg(
        **{"Total Leads": "size", "Total Students": "sum"}
    )
    results = (
        counselors.merge(stats, left_on="Counselor", right_index=True, how="left")
        .sort_values("Conversion_Probability", ascending=False)
        .reset_index(drop=True)
    )

    best = results.iloc[0]
    st.success(
        f"Best Counselor Assigned: {best['Counselor']} (Level: {best['Counselor Level']})"
    )
    st.write(f"Predicted Conversion Probability: {best['Conversion_Probability']:.2f}")

    st.subheader("All Counselor Predictions")
    st.dataframe(results)
