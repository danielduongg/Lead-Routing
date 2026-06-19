# Lead Routing — Lead-to-Counselor Assignment

A Streamlit app that recommends the counselor most likely to convert a given
lead into an enrolled student. It trains a Random Forest on historical lead data
and, for a new lead, scores every counselor and ranks them by predicted
conversion probability.

**Live demo:** https://data-science-projects-fmxd5igvkmemcryce7jbry.streamlit.app/

## Apps

| File | Description |
| --- | --- |
| `lead_assignment_app.py` | Baseline app. Inputs: College, Program Level, Program of Study. |
| `lead_assignment_app2.py` | Enhanced app. Adds Lead Source & State as inputs, a tuned model, and an on-screen evaluation (classification report, ROC AUC, confusion matrix). |

## Data

`Lead Conversion Data.xlsx` (~300k rows). The model target `Converted` is derived
from the `Record Type` column (`Student` → 1, otherwise 0). Data is loaded with
the fast `calamine` engine and cached, and the trained model is cached across
reruns, so the app only does the heavy work once.

## Run locally

```bash
git clone https://github.com/danielduongg/Lead-Routing.git
cd Lead-Routing

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
streamlit run lead_assignment_app.py      # or lead_assignment_app2.py
```

The app opens at http://localhost:8501.

## How it works

1. Load and cache the historical lead data.
2. Train a one-hot-encoded Random Forest pipeline (cached) to predict conversion.
3. For a new lead, build one candidate row per counselor, score them in a single
   batch, and rank by predicted conversion probability.

## Notes

- A Codespaces dev container is included (`.devcontainer/`) that installs
  requirements and launches the baseline app automatically.
