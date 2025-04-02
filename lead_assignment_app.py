import pandas as pd 
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load and preprocess the data
df = pd.read_excel("Lead Conversion Data.xlsx")
df['Converted'] = df['Record Status'].apply(lambda x: 1 if str(x).strip().lower() == "started" else 0)

features = ['State', 'College', 'Program Level', 'Program of Study', 'Counselor', 'Counselor Level']
df_model = df[features + ['Converted']].dropna()
X = df_model[features]
y = df_model['Converted']

categorical_features = features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Define Streamlit UI
st.title("Lead to Counselor Assignment App")
st.write("Input new lead details to automatically assign the best counselor.")

state = st.selectbox("State", df['State'].dropna().unique())
college = st.selectbox("College", df['College'].dropna().unique())
program_level = st.selectbox("Program Level", df['Program Level'].dropna().unique())
program_of_study = st.selectbox("Program of Study", df['Program of Study'].dropna().unique())

if st.button("Assign Counselor"):
    new_lead_input = {
        'State': state,
        'College': college,
        'Program Level': program_level,
        'Program of Study': program_of_study
    }

    counselors_df = df[['Counselor', 'Counselor Level']].dropna().drop_duplicates()
    predictions = []
    for _, row in counselors_df.iterrows():
        lead_input = new_lead_input.copy()
        lead_input['Counselor'] = row['Counselor']
        lead_input['Counselor Level'] = row['Counselor Level']
        candidate_df = pd.DataFrame([lead_input])
        prob = pipeline.predict_proba(candidate_df)[0, 1]
        predictions.append({
            'Counselor': row['Counselor'],
            'Counselor Level': row['Counselor Level'],
            'Conversion_Probability': prob
        })

    predictions_df = pd.DataFrame(predictions)
    best_assignment = predictions_df.sort_values(by='Conversion_Probability', ascending=False).iloc[0]

    st.success(f"Best Counselor Assigned: {best_assignment['Counselor']} (Level: {best_assignment['Counselor Level']})")
    st.write(f"Predicted Conversion Probability: {best_assignment['Conversion_Probability']:.2f}")

    st.subheader("All Counselor Predictions")
    st.dataframe(predictions_df.sort_values(by='Conversion_Probability', ascending=False))
