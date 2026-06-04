import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# 1. Page Title
st.title("Patient Heart Disease Predictor")


# 2. Load Data and Train Model on the Fly
@st.cache_resource
def get_trained_model():
    # Load your dataset
    df = pd.read_csv("data/heart.csv")

    # Drop target column to separate features and labels
    X = df.drop("output", axis=1)
    y = df["output"]

    # Slice out only the features you are gathering from sliders
    # Note: A model must be trained on the exact same columns it predicts
    X_subset = X[["age", "chol", "thalachh"]]

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_subset, y)
    return clf


# Initialize the model
try:
    model = get_trained_model()
except Exception as e:
    st.error(f"Could not train model. Verify data/heart.csv exists. Error: {e}")
    st.stop()

# 3. User Interface Inputs
age = st.slider("Age", 20, 90, 50)
chol = st.slider("Cholesterol", 100, 600, 200)
thalachh = st.slider("Max Heart Rate", 60, 220, 150)

# 4. Handle Prediction
if st.button("Predict"):
    # Match the column names exactly to X_subset
    sample = pd.DataFrame(
        {"age": [age], "chol": [chol], "thalachh": [thalachh]}
    )

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
