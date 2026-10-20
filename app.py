
import os
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Program",
    "PlacementPredict_Cleaned.csv"
)

CATEGORICAL_FEATURES = [
    "Gender", "City", "CollegeTier", "Stream",
    "Specialisation", "Hostel", "HistoryOfBacklogs"
]

NUMERIC_FEATURES = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "AttendancePercent", "Internships", "Projects", "Workshops",
    "Certifications", "Publications", "AptitudeTestScore",
    "SoftSkillsRating", "CodingTestScore", "MockInterviewScore",
    "ExtraCurricular"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["PlacementStatus"]).copy()

X = df[FEATURES].copy()
y = pd.to_numeric(df["PlacementStatus"], errors="coerce")

valid = y.isin([0, 1])
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

choices = {}

for col in CATEGORICAL_FEATURES:
    values = df[col].dropna().astype(str).unique().tolist()
    choices[col] = sorted(values)

NUMERIC_DEFAULTS = {
    "SGPA_Sem1": 7.0,
    "SGPA_Sem2": 7.0,
    "SGPA_Sem3": 7.0,
    "SGPA_Sem4": 7.0,
    "SGPA_Sem5": 7.0,
    "SGPA_Sem6": 7.0,
    "SGPA_Sem7": 7.0,
    "SGPA_Sem8": 7.0,
    "AttendancePercent": 80.0,
    "Internships": 1,
    "Projects": 2,
    "Workshops": 2,
    "Certifications": 2,
    "Publications": 0,
    "AptitudeTestScore": 65.0,
    "SoftSkillsRating": 3.0,
    "CodingTestScore": 60.0,
    "MockInterviewScore": 60.0,
    "ExtraCurricular": 1
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Placement Prediction</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 28px 14px;
    font-family: Arial, sans-serif;
    color: #172554;
    background: linear-gradient(135deg, #e0f2fe, #eef2ff);
    min-height: 100vh;
}

.container {
    max-width: 1000px;
    margin: auto;
}

header {
    text-align: center;
    padding: 24px 12px;
}

header h1 {
    margin: 0 0 12px;
    font-size: clamp(26px, 4vw, 38px);
}

header p {
    color: #475569;
    line-height: 1.6;
}

.panel {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 35px #1e3a8a16;
}

.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
}

.field label {
    display: block;
    font-weight: 600;
    margin-bottom: 8px;
    font-size: 14px;
}

input, select {
    width: 100%;
    padding: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    background: white;
    font-size: 15px;
}

button {
    width: 100%;
    margin-top: 24px;
    padding: 15px;
    border: none;
    border-radius: 10px;
    background: #1d4ed8;
    color: white;
    font-weight: bold;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #1e40af;
}

.result {
    margin-top: 22px;
    padding: 20px;
    border-radius: 12px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    text-align: center;
}

.note {
    font-size: 13px;
    color: #64748b;
    line-height: 1.6;
    margin-top: 22px;
}

@media (max-width: 600px) {
    .grid {
        grid-template-columns: 1fr;
    }

    .panel {
        padding: 17px;
    }
}
</style>
</head>

<body>
<div class="container">

<header>
    <h1>Student Placement Prediction</h1>
    <p>
        Explore a machine-learning prediction based on academic
        performance, skills and student activities.
    </p>
</header>

<div class="panel">

<form method="POST">
<div class="grid">

{% for col in categorical %}
<div class="field">
    <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>

    <select id="{{ col }}" name="{{ col }}" required>
    {% for option in choices[col] %}
        <option value="{{ option }}"
        {% if request.form.get(col, choices[col][0]) == option %}
        selected
        {% endif %}>
            {{ option }}
        </option>
    {% endfor %}
    </select>
</div>
{% endfor %}

{% for col in numeric %}
<div class="field">
    <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>

    <input
        id="{{ col }}"
        name="{{ col }}"
        type="number"
        step="any"
        min="0"
        required
        value="{{ request.form.get(col, defaults[col]) }}"
    >
</div>
{% endfor %}

</div>

<button type="submit">Predict Placement Status</button>
</form>

{% if result is not none %}
<div class="result" role="status">
    <h2>{{ result }}</h2>

    {% if probability is not none %}
    <p>
        Model-estimated probability of placement:
        <strong>{{ probability }}%</strong>
    </p>
    {% endif %}

    <p>Test-set accuracy: {{ accuracy }}%</p>
</div>
{% endif %}

<p class="note">
    This is an educational machine-learning demonstration.
    Predictions are estimates, not guarantees of employment.
    Results depend on the dataset and model quality.
</p>

</div>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":
        try:
            row = {}

            for col in CATEGORICAL_FEATURES:
                value = request.form.get(col, "")

                if value not in choices[col]:
                    raise ValueError(
                        "Please select a valid category."
                    )

                row[col] = value

            for col in NUMERIC_FEATURES:
                value = float(request.form[col])

                if not pd.notna(value) or value < 0:
                    raise ValueError(
                        "Enter valid non-negative values."
                    )

                row[col] = value

            sample = pd.DataFrame(
                [row],
                columns=FEATURES
            )

            prediction = int(model.predict(sample)[0])
            probabilities = model.predict_proba(sample)[0]

            positive_index = list(model.classes_).index(1)

            probability = round(
                float(probabilities[positive_index]) * 100,
                2
            )

            if prediction == 1:
                result = "Likely to be placed"
            else:
                result = "Unlikely to be placed"

        except (ValueError, KeyError):
            result = "Please check your inputs and try again."
            probability = None

    return render_template_string(
        HTML,
        categorical=CATEGORICAL_FEATURES,
        numeric=NUMERIC_FEATURES,
        choices=choices,
        defaults=NUMERIC_DEFAULTS,
        result=result,
        probability=probability,
        accuracy=round(accuracy * 100, 2),
        request=request
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
