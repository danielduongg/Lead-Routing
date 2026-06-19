"""Lead-to-Counselor assignment app (enhanced).

Adds more predictive features, a tuned Random Forest, and an on-screen model
evaluation (classification report, ROC AUC, confusion matrix) on top of the
baseline assignment workflow.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "Lead Conversion Data.xlsx"
# Richer feature set than the baseline app (all columns that exist in the data).
FEATURES = [
    "Lead Source",
    "State",
    "College",
    "Program Level",
    "Program of Study",
    "Counselor",
    "Counselor Level",
]
INPUT_FEATURES = ["Lead Source", "State", "College", "Program Level", "Program of Study"]


@st.cache_data(show_spinner="Loading lead data…")
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, engine="calamine")
    except Exception:
        df = pd.read_excel(path)
    df["Converted"] = (
        df["Record Type"].astype(str).str.strip().str.lower() == "student"
    ).astype(int)
    return df


@st.cache_resource(show_spinner="Training model…")
def train_model(path: str = DATA_PATH):
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
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


df = load_data()
pipeline, X_test, y_test = train_model()

st.title("Lead to Counselor Assignment App")
st.write("Input new lead details to automatically assign the best counselor.")

# ---- Model evaluation ----
with st.expander("Model Evaluation", expanded=False):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Lead", "Student"], yticklabels=["Lead", "Student"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---- Assignment UI ----
selections = {
    name: st.selectbox(name, sorted(df[name].dropna().unique())) for name in INPUT_FEATURES
}

if st.button("Assign Counselor"):
    counselors = (
        df[["Counselor", "Counselor Level"]].dropna().drop_duplicates().reset_index(drop=True)
    )
    candidates = counselors.copy()
    for name, value in selections.items():
        candidates[name] = value

    counselors["Conversion_Probability"] = pipeline.predict_proba(candidates[FEATURES])[:, 1]
    results = counselors.sort_values(
        "Conversion_Probability", ascending=False
    ).reset_index(drop=True)

    best = results.iloc[0]
    st.success(
        f"Best Counselor Assigned: {best['Counselor']} (Level: {best['Counselor Level']})"
    )
    st.write(f"Predicted Conversion Probability: {best['Conversion_Probability']:.2f}")

    st.subheader("All Counselor Predictions")
    st.dataframe(results)
