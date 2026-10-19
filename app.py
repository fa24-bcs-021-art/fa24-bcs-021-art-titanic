import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

NUM = ["Age", "SibSp", "Parch", "Fare"]
CAT = ["Pclass", "Sex", "Embarked"]


@st.cache_resource(show_spinner="Training model...")
def load_model():
    df = pd.read_csv("titanic.csv")
    X = df[NUM + CAT].copy()
    X["Pclass"] = X["Pclass"].astype(str)
    y = df["Survived"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                                     random_state=42))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    model.fit(X, y)  # refit on all data for deployment
    return model, acc


model, acc = load_model()

st.title("🚢 Titanic Survival Predictor")
st.caption(f"Random Forest model · hold-out accuracy: {acc:.1%}")

with st.form("passenger"):
    c1, c2 = st.columns(2)
    pclass = c1.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = c2.selectbox("Sex", ["male", "female"])
    age = c1.number_input("Age", 0.0, 100.0, 30.0, step=1.0)
    fare = c2.number_input("Fare (£)", 0.0, 600.0, 32.0, step=1.0)
    sibsp = c1.number_input("Siblings / spouses aboard", 0, 10, 0)
    parch = c2.number_input("Parents / children aboard", 0, 10, 0)
    embarked = st.selectbox("Port of embarkation",
                            ["S", "C", "Q"],
                            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    submit = st.form_submit_button("Predict", use_container_width=True)

if submit:
    row = pd.DataFrame([{"Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
                         "Pclass": str(pclass), "Sex": sex, "Embarked": embarked}])
    p = model.predict_proba(row)[0][1]
    if p >= 0.5:
        st.success(f"✅ Likely to SURVIVE — probability {p:.1%}")
    else:
        st.error(f"❌ Unlikely to survive — survival probability {p:.1%}")
    st.progress(float(p))

with st.expander("About"):
    st.write("AIC354 Machine Learning Fundamentals · Assignment 2 · Task 1 (FA24-BCS-021). "
             "Trained on the classic Titanic dataset (891 passengers).")
