import streamlit as st
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Predictor")

MODEL_PATH = Path(__file__).parent / "titanic_model.pkl"

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. If you have a trained model, place `titanic_model.pkl` in this folder.`")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.markdown("Use the controls below to enter passenger features and get a survival prediction.")

# Inputs
pclass = st.selectbox("Pclass", [1, 2, 3], index=1)
sex = st.selectbox("Sex", ["male", "female"], index=0)
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.2)
embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)

# Simple preprocessing - adapt to your model's expected feature order
sex_map = {"male": 0, "female": 1}
emb_map = {"S": 0, "C": 1, "Q": 2}

features = [pclass, sex_map.get(sex, 0), age, sibsp, parch, fare, emb_map.get(embarked, 0)]
X = np.array(features).reshape(1, -1)

st.write("---")

if st.button("Predict"):
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
            st.write(f"Predicted survival probability: {prob:.3f}")
            if prob >= 0.5:
                st.success("The model predicts the passenger would likely survive.")
            else:
                st.warning("The model predicts the passenger would likely not survive.")
        else:
            pred = model.predict(X)[0]
            st.write(f"Predicted label: {pred}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
