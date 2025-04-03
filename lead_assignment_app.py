import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load and preprocess the data
df = pd.read_excel("Lead Conversion Data.xlsx")
df['Converted'] = df['Record Type'].apply(lambda x: 1 if str(x).strip().lower() == "student" else 0)

features = ['State', 'College', 'Program Level', 'Program of Study', 'Counselor']
df_model = df[features + ['Converted']].dropna()

# Compute conversion rates per counselor
conversion_rates = df_model.groupby('Counselor')['Converted'].agg(['count', 'sum']).reset_index()
conversion_rates['Conversion Rate'] = conversion_rates['sum'] / conversion_rates['count']
conversion_rates = conversion_rates.sort_values(by='Conversion Rate', ascending=False)

# Prepare ML model to find counselor for new lead
X = df_model[['State', 'College', 'Program Level', 'Program of Study', 'Counselor']]
y = df_model['Converted']

categorical_features = ['State', 'College', 'Program Level', 'Program of Study', 'Counselor']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
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

# Streamlit app UI
st.title("Lead to Counselor Assignment App")
st.write("Assign a new lead to the counselor with the highest conversion probability.")

state = st.selectbox("State", df['State'].dropna().unique())
college = st.selectbox("College", df['College'].dropna().unique())
program_level = st.selectbox("Program Level", df['Program Level'].dropna().unique())
program_of_study = st.selectbox("Program of Study", df['Program of Study'].dropna().unique())

if st.button("Assign Best Counselor"):
    input_lead = {
        'State': state,
        'College': college,
        'Program Level': program_level,
        'Program of Study': program_of_study
    }

    predictions = []
    counselors = df['Counselor'].dropna().unique()
    for counselor in counselors:
        lead_with_counselor = input_lead.copy()
        lead_with_counselor['Counselor'] = counselor
        input_df = pd.DataFrame([lead_with_counselor])
        prob = pipeline.predict_proba(input_df)[0, 1]
        predictions.append({
            'Counselor': counselor,
            'Conversion_Probability': prob
        })

    predictions_df = pd.DataFrame(predictions)
    best_counselor = predictions_df.sort_values(by='Conversion_Probability', ascending=False).iloc[0]

    st.success(f"Assigned Counselor: {best_counselor['Counselor']}")
    st.write(f"Predicted Conversion Probability: {best_counselor['Conversion_Probability']:.2f}")

    st.subheader("All Counselor Predictions")
    st.dataframe(predictions_df.sort_values(by='Conversion_Probability', ascending=False))
