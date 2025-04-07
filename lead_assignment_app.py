import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Lead Conversion Data.xlsx", sheet_name="Sheet1")
    return df

df = load_data()

# Calculate conversion rate per counselor per college
converted = df[df['Record Type'] == 'Student']
total = df.groupby(['College', 'Counselor']).size().reset_index(name='Total Leads')
starts = converted.groupby(['College', 'Counselor']).size().reset_index(name='Converted Leads')
rates = pd.merge(total, starts, on=['College', 'Counselor'], how='left')
rates['Converted Leads'] = rates['Converted Leads'].fillna(0)
rates['Conversion Rate'] = rates['Converted Leads'] / rates['Total Leads']

# Get the best counselor by college
best_counselors = rates.sort_values(by=['College', 'Conversion Rate'], ascending=[True, False])
best_by_college = best_counselors.drop_duplicates(subset='College', keep='first')
best_by_college = best_by_college.set_index('College')

# Streamlit UI
st.title("Lead Assignment by Conversion Rate")

college_options = sorted(df['College'].dropna().unique())
college = st.selectbox("Select College", college_options)

if college in best_by_college.index:
    best = best_by_college.loc[college]
    st.success(f"Assign this lead to: {best['Counselor']} (Conversion Rate: {best['Conversion Rate']:.2%})")
else:
    st.warning("No data available for this college.")
