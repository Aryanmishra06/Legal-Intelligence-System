from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

 
DATA_PATH = os.path.join(os.path.dirname(__file__), "Court_dataset.csv")

# -----------------------------
# GLOBAL CACHE
# -----------------------------
df_global = None
vectorizer_global = None
matrix_global = None
model_global = None
le_type_global = None
le_cat_global = None
le_out_global = None

# -----------------------------
# LOAD DATASET
# -----------------------------
def load_data():
    global df_global
    if df_global is not None:
        return df_global
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        df_global = pd.DataFrame(columns=["name","case_type","case_category","judge","status","outcome","year"])
        return df_global

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.fillna("")

    
    if "year" not in df.columns:
        df["year"] = df["name"].str.extract(r'(\d{4})$').fillna("")
    else:
        df["year"] = df["name"].str.extract(r'(\d{4})$').fillna("")

    df["name"] = df["name"].astype(str)
    df["case_type"] = df["case_type"].astype(str)
    df["case_category"] = df["case_category"].astype(str)
    df["judge"] = df["judge"].astype(str) if "judge" in df.columns else ""
    df["status"] = df["status"].astype(str) if "status" in df.columns else ""
    df["outcome"] = df["outcome"].astype(str) if "outcome" in df.columns else ""

    
    df["combined"] = (
        df["name"] + " " +
        df["case_type"] + " " + df["case_type"] + " " +  # weight case_type
        df["case_category"] + " " +
        df["judge"] + " " +
        df["status"] + " " +
        df["outcome"] + " " +
        df["year"]
    )

    df_global = df
    print(f"[INFO] Dataset loaded: {len(df)} rows")
    return df

# -----------------------------
# SEARCH MODEL
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
MATRIX_PATH = os.path.join(os.path.dirname(__file__), "matrix.pkl")

def load_or_train_search_model(df):
    global vectorizer_global, matrix_global
    if vectorizer_global is not None:
        return vectorizer_global, matrix_global

   
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    matrix = vectorizer.fit_transform(df["combined"])

    vectorizer_global = vectorizer
    matrix_global = matrix
    return vectorizer, matrix

# -----------------------------
# SMART SEARCH
# -----------------------------
def ml_search(query, df, vectorizer, matrix):
    if not query.strip():
        return df.head(20)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, matrix).flatten()

    
    top_indices = similarity.argsort()[-50:][::-1]
    results = df.iloc[top_indices].copy()
    results["score"] = similarity[top_indices]
    results = results[results["score"] > 0]  # filter zero-score results
    return results

# -----------------------------
# PREDICTION MODEL
# -----------------------------
def train_prediction_model(df):
    global model_global, le_type_global, le_cat_global, le_out_global
    if model_global is not None:
        return model_global, le_type_global, le_cat_global, le_out_global

    df = df.copy()
    df = df[df["outcome"].notna() & (df["outcome"] != "") & (df["outcome"] != "nan")]

    le_type = LabelEncoder()
    le_cat = LabelEncoder()
    le_out = LabelEncoder()

    df["type_enc"] = le_type.fit_transform(df["case_type"])
    df["cat_enc"] = le_cat.fit_transform(df["case_category"])
    df["out_enc"] = le_out.fit_transform(df["outcome"])

    X = df[["type_enc","cat_enc"]]
    y = df["out_enc"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    model_global = model
    le_type_global = le_type
    le_cat_global = le_cat
    le_out_global = le_out

    print("[INFO] Prediction model trained.")
    return model, le_type, le_cat, le_out

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/live_search")
def live_search():
    query = request.args.get("q", "").strip()
    df = load_data()
    vectorizer, matrix = load_or_train_search_model(df)
    results = ml_search(query, df, vectorizer, matrix)

   
    cols = ["name", "case_type", "case_category", "judge", "status", "outcome", "year", "score"]
    cols = [c for c in cols if c in results.columns]
    output = results[cols].copy()
    output["year"] = output["name"].str.extract(r'(\d{4})$').fillna("—")
    return jsonify(output.to_dict(orient="records"))

@app.route("/predict", methods=["POST"])
def predict():
    case_type = request.form.get("case_type", "").strip()
    case_category = request.form.get("case_category", "").strip()
    if not case_type or not case_category:
        return jsonify({"prediction": "Enter valid inputs"})
    df = load_data()
    model, le_type, le_cat, le_out = train_prediction_model(df)
    try:
        
        case_type = case_type.strip().title()
        case_category = case_category.strip().title()

        type_enc = le_type.transform([case_type])[0]
        cat_enc = le_cat.transform([case_category])[0]
        pred = model.predict([[type_enc, cat_enc]])[0]
        result = le_out.inverse_transform([pred])[0]
        return jsonify({"prediction": result})
    except ValueError as e:
        print(f"[PREDICT ERROR] {e}")
        # Valid options bhi bata do
        valid_types = list(le_type.classes_)
        valid_cats = list(le_cat.classes_)
        return jsonify({
            "prediction": "Unknown",
            "valid_types": valid_types,
            "valid_categories": valid_cats
        })

@app.route("/dashboard")
def dashboard():
    df = load_data()
    total = len(df)
    types = df["case_type"].value_counts().head(5).to_dict() if "case_type" in df.columns else {}
    categories = df["case_category"].value_counts().head(5).to_dict() if "case_category" in df.columns else {}
    pending = len(df[df["status"].str.lower() == "pending"]) if "status" in df.columns else 0
    closed = len(df[df["status"].str.lower() == "closed"]) if "status" in df.columns else 0
    judges = df["judge"].value_counts().head(10).to_dict() if "judge" in df.columns else {}
    return render_template("dashboard.html", total=total, types=types, categories=categories,
                           judges=judges, pending=pending, closed=closed)

 
@app.route("/get_options")
def get_options():
    df = load_data()
    types = sorted(df["case_type"].unique().tolist())
    categories = sorted(df["case_category"].unique().tolist())
    return jsonify({"case_types": types, "case_categories": categories})

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)